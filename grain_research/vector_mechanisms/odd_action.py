#!/usr/bin/env python3
"""Bidirectional adjoints and reference-only reflection-odd functional readers."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.centered_adjoint import (
    center_groups,
    integrate_nodes,
)

FAMILIES = ("position", "state_even", "combined")


def bidirectional_account(displacement, positive, negative, labels):
    """Integrate matching rules and separate odd/even effects, retaining rounding error."""
    if sorted(positive) != sorted(negative):
        raise ValueError("Positive and negative paths require matched quadrature nodes")
    gp, gm = integrate_nodes(positive), integrate_nodes(negative)
    d = np.asarray(displacement, float)
    if d.shape != gp.shape or gm.shape != gp.shape:
        raise ValueError("Matched token displacement and adjoints required")
    ge, go = (gp + gm) / 2, (gp - gm) / 2
    odd, even = float(np.sum(d * ge)), float(np.sum(d * go))
    dc = center_groups(d, labels)
    return {
        "integrated_even": ge,
        "integrated_odd": go,
        "odd_action": odd,
        "even_action": even,
        "plus_action": float(np.sum(d * gp)),
        "odd_centering_remainder": odd - float(np.sum(dc * center_groups(ge, labels))),
        "even_centering_remainder": even
        - float(np.sum(dc * center_groups(go, labels))),
    }


def even_features(displacement, labels, projection, state_scale):
    """Build frozen even position/state coordinates, centered within each comment."""
    d = center_groups(displacement, labels)
    p, scale = np.asarray(projection, float), np.asarray(state_scale, float)
    if (
        p.shape != (d.shape[1], len(scale))
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Finite positive fixed projection scales required")
    s = np.arange(len(d)) / max(len(d) - 1, 1)
    t = np.empty(len(d))
    for group in np.unique(labels):
        ix = np.flatnonzero(labels == group)
        t[ix] = np.arange(len(ix)) / max(len(ix) - 1, 1)
    position = np.column_stack(
        [
            np.cos(np.pi * k * s) * np.cos(np.pi * l * t)
            for k in range(4)
            for l in range(8)
        ]
    )
    position = center_groups(position, labels)
    state = center_groups((d @ p / scale) ** 2, labels)
    return {
        "position": position,
        "state_even": state,
        "combined": np.column_stack([position, state]),
    }


def fit_odd_fields(
    packets, projection, state_scale, field_alpha=1.0, action_alpha=0.01
):
    """Fit all predeclared fields and action weights using reference authors only."""
    refs = [p for p in packets if p["role"] == "reference"]
    if (
        not refs
        or min(field_alpha, action_alpha) <= 0
        or any(p["role"] not in {"reference", "evaluation"} for p in packets)
    ):
        raise ValueError("Explicit author roles and positive penalties required")
    features = {k: [] for k in FAMILIES}
    ds, gs, ws = [], [], []
    for packet in refs:
        d = center_groups(packet["displacement"], packet["labels"])
        g = len(d) * center_groups(packet["integrated_even"], packet["labels"])
        ds.append(d)
        gs.append(g)
        ws.append(np.full(len(d), 1 / (len(refs) * len(d))))
        for name, phi in even_features(
            d, packet["labels"], projection, state_scale
        ).items():
            features[name].append(phi)
    d, gamma, w = map(np.concatenate, (ds, gs, ws))
    target = np.sum(d * gamma, axis=1)
    model = {"projection": projection, "state_scale": state_scale}
    for name in FAMILIES:
        phi = np.concatenate(features[name])
        scale = np.sqrt(np.sum(w[:, None] * phi**2, axis=0))
        scale = np.where(scale > 1e-10, scale, 1.0)
        x = phi / scale
        matrix = np.linalg.solve(
            x.T @ (w[:, None] * x) + field_alpha * np.eye(x.shape[1]),
            x.T @ (w[:, None] * gamma),
        )
        action_basis = x * (d @ matrix.T)
        action_scale = np.sqrt(np.sum(w[:, None] * action_basis**2, axis=0))
        action_scale = np.where(action_scale > 1e-12, action_scale, 1.0)
        b = action_basis / action_scale
        gram = b.T @ (w[:, None] * b)
        rhs = b.T @ (w * target)
        prior = action_scale.copy()
        theta = prior + np.linalg.solve(
            gram + action_alpha * np.eye(len(gram)), rhs - gram @ prior
        )
        model[f"{name}_feature_scale"] = scale
        model[f"{name}_matrix"] = matrix
        model[f"{name}_action_weights"] = theta / action_scale
        model[f"{name}_action_scale"] = action_scale
    return model


def odd_token_readouts(displacement, labels, model):
    """Evaluate frozen odd functions without neural suffixes or scorer gradients."""
    d = center_groups(displacement, labels)
    features = even_features(d, labels, model["projection"], model["state_scale"])
    result = {}
    for name, phi in features.items():
        b = (phi / model[f"{name}_feature_scale"]) * (d @ model[f"{name}_matrix"].T)
        result[f"{name}_field"] = b.sum(axis=1)
        result[f"{name}_action"] = b @ model[f"{name}_action_weights"]
    return result
