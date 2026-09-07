#!/usr/bin/env python3
"""Leaf-head differentiation and finite shared-response geometry."""

from __future__ import annotations

import numpy as np
import torch


def projection_gradient(projection, donor, forward_score):
    """Differentiate a scalar closure through one final-token projection input.

    All other positions are detached constants. The caller supplies a frozen
    model and a scalar, differentiable scoring closure; hooks are always removed.
    """
    if (
        donor.ndim != 1
        or not donor.is_floating_point()
        or not torch.isfinite(donor).all()
    ):
        raise ValueError("Finite floating head vector required")
    leaf = donor.detach().clone().requires_grad_(True)
    calls = 0

    def replace(_module, args):
        nonlocal calls
        calls += 1
        state = args[0].detach().clone()
        if state.ndim != 3 or state.shape[0] != 1 or state[0, -1].shape != leaf.shape:
            raise ValueError("One compact sequence and compatible head required")
        if state.dtype != leaf.dtype or state.device != leaf.device:
            raise ValueError("Donor dtype/device must match the actual head")
        state[0, -1] = leaf
        return (state, *args[1:])

    handle = projection.register_forward_pre_hook(replace)
    try:
        with torch.enable_grad():
            score = forward_score()
            if calls != 1 or score.ndim != 0 or not torch.isfinite(score):
                raise ValueError("One projection call and finite scalar score required")
            (gradient,) = torch.autograd.grad(score, leaf)
        if not torch.isfinite(gradient).all():
            raise ValueError("Finite head gradient required")
        return float(score.detach().cpu()), gradient.detach().double().cpu().numpy()
    finally:
        handle.remove()


def simpson_gradient(nodes):
    """Integrate three/five equally spaced gradient vectors on a unit path."""
    if set(nodes) == {0.0, 0.5, 1.0}:
        times, weights = [0.0, 0.5, 1.0], np.array([1, 4, 1]) / 6
    elif set(nodes) == {0.0, 0.25, 0.5, 0.75, 1.0}:
        times, weights = [0.0, 0.25, 0.5, 0.75, 1.0], np.array([1, 4, 2, 4, 1]) / 12
    else:
        raise ValueError("Complete three/five-node Simpson rule required")
    values = np.asarray([nodes[t] for t in times], float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Compatible finite gradient vectors required")
    return weights @ values


def leave_author_out_geometry(gradients, displacements, ranks=(1, 2, 4)):
    """Separate no-test-gradient prediction from test-gradient subspace diagnosis."""
    g, d = np.asarray(gradients, float), np.asarray(displacements, float)
    if g.ndim != 2 or g.shape != d.shape or len(g) < 3 or not np.isfinite([g, d]).all():
        raise ValueError("Paired finite author gradient/displacement matrices required")
    if any(
        not isinstance(rank, int) or rank < 1 or rank > min(len(g) - 1, g.shape[1])
        for rank in ranks
    ):
        raise ValueError("Valid fixed ranks required")
    rows = []
    for i in range(len(g)):
        train = np.arange(len(g)) != i
        shared = g[train].mean(0)
        row = {
            "loo_mean_prediction": float(shared @ d[i]),
            "self_integral_prediction": float(g[i] @ d[i]),
        }
        _, _, vh = np.linalg.svd(g[train], full_matrices=False)
        for rank in ranks:
            projected = vh[:rank].T @ (vh[:rank] @ g[i])
            row[f"rank{rank}_diagnostic"] = float(projected @ d[i])
            row[f"rank{rank}_gradient_error"] = (
                float(np.sum((g[i] - projected) ** 2) / np.sum(g[i] ** 2))
                if np.any(g[i])
                else None
            )
        rows.append(row)
    return rows
