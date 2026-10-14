#!/usr/bin/env python3
"""Neutral mixed-derivative binding and person-specific response accounts."""

from __future__ import annotations

import numpy as np


def gaussian_basis(dimension, width, seed):
    """Return a deterministic orthonormal sketch without selecting its width."""
    if not 0 < width <= dimension:
        raise ValueError("Sketch width must fit the hidden dimension")
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, width)))[0]


def compressed_svd(range_values, transpose_values, rank):
    """Factor Q^T H from H Omega and H^T Q; do not imply a full Hessian SVD."""
    y, t = np.asarray(range_values, float), np.asarray(transpose_values, float)
    if (
        y.ndim != 2
        or t.shape != y.shape
        or not np.isfinite(y).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError("Finite compatible Hessian sketches required")
    if not 0 < rank <= y.shape[1]:
        raise ValueError("Rank outside fixed sketch support")
    q = np.linalg.qr(y)[0]
    small_u, singular, vt = np.linalg.svd(t.T, full_matrices=False)
    return q @ small_u[:, :rank], singular, vt[:rank].T


def swap_delta(left, right, basis, match_norm=None):
    """Project a within-context slot difference, optionally matching edit energy."""
    a, b, u = (
        np.asarray(left, float),
        np.asarray(right, float),
        np.asarray(basis, float),
    )
    if a.shape != b.shape or a.ndim != 1 or u.ndim != 2 or u.shape[0] != len(a):
        raise ValueError("Aligned vector slots and column basis required")
    if not np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-6):
        raise ValueError("Intervention basis must be orthonormal")
    delta = u @ (u.T @ (b - a))
    if match_norm is not None:
        size = np.linalg.norm(delta)
        if size == 0 and match_norm != 0:
            raise ValueError("Random control cannot match a nonzero action")
        if size:
            delta *= match_norm / size
    return delta


def mixed_product(
    function, dimension, vectors, transpose=False, device="cpu", checkpoint=None
):
    """Differentiate a fixed scalar function; never optimize model parameters."""
    import torch

    x = torch.zeros(dimension, device=device, requires_grad=True)
    y = torch.zeros(dimension, device=device, requires_grad=True)
    scalar = function(x, y)
    if scalar.ndim != 0 or not torch.isfinite(scalar):
        raise ValueError("Finite scalar retrieval functional required")
    gx, gy = torch.autograd.grad(scalar, (x, y), create_graph=True)
    gradient, variable = (gx, y) if transpose else (gy, x)
    products = []
    for column in np.asarray(vectors, float).T:
        v = torch.tensor(column, device=device, dtype=x.dtype)
        product = torch.autograd.grad(gradient @ v, variable, retain_graph=True)[0]
        products.append(product.detach().cpu().numpy())
        if checkpoint is not None:
            checkpoint()
    return float(scalar.detach().cpu()), np.stack(products, axis=1)


def routing_accounts(baseline, edited, natural_swapped):
    """Separate person-differential, common and untouched-person effects."""
    b, e, n = (np.asarray(v, float) for v in (baseline, edited, natural_swapped))
    if b.shape != e.shape or b.shape != n.shape or b.ndim != 2 or b.shape[0] != 3:
        raise ValueError("Three people with aligned target readouts required")
    actual = (e[0] - e[1]) - (b[0] - b[1])
    expected = (n[0] - n[1]) - (b[0] - b[1])
    norm = np.linalg.norm(actual) * np.linalg.norm(expected)
    energy = expected @ expected
    return {
        "person_difference_effect": actual.tolist(),
        "natural_assignment_effect": expected.tolist(),
        "towards_natural_fraction": float(actual @ expected / energy)
        if energy
        else None,
        "direction_cosine": float(actual @ expected / norm) if norm else None,
        "common_effect": ((e[:2] - b[:2]).mean(0)).tolist(),
        "untouched_effect": (e[2] - b[2]).tolist(),
        "differential_norm": float(np.linalg.norm(actual)),
        "common_norm": float(np.linalg.norm((e[:2] - b[:2]).mean(0))),
        "untouched_norm": float(np.linalg.norm(e[2] - b[2])),
    }
