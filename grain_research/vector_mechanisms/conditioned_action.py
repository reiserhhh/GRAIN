#!/usr/bin/env python3
"""Conditional tensor-product action readers with a frozen reference population."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.centered_adjoint import center_groups

CONDITIONS = {
    "background": np.arange(8),
    "layout": np.arange(8, 12),
    "combined": np.arange(12),
}


def token_conditions(body, labels, projection):
    """Form numeric comment-background and ordered-layout conditions without text or IDs."""
    h, groups, p = (
        np.asarray(body, float),
        np.asarray(labels),
        np.asarray(projection, float),
    )
    if (
        h.ndim != 2
        or groups.shape != (len(h),)
        or p.shape != (h.shape[1], 8)
        or not np.isfinite(h).all()
    ):
        raise ValueError(
            "Finite body states, comment labels and fixed eight-direction projection required"
        )
    unique = np.unique(groups)
    z = np.empty((len(h), 12))
    z[:, 9] = np.arange(len(h)) / max(len(h) - 1, 1)
    for i, group in enumerate(unique):
        indices = np.flatnonzero(groups == group)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("Comment token groups must be contiguous in body order")
        z[indices, :8] = h[indices].mean(0) @ p
        z[indices, 8] = np.log1p(len(indices))
        z[indices, 10] = np.arange(len(indices)) / max(len(indices) - 1, 1)
        z[indices, 11] = i / max(len(unique) - 1, 1)
    return z


def fit_conditioned_fields(packets, base, projection, alpha=0.01):
    """Fit three predeclared tensor products using only reference-author token actions."""
    refs = [p for p in packets if p["role"] == "reference"]
    if (
        not refs
        or alpha <= 0
        or any(p["role"] not in {"reference", "evaluation"} for p in packets)
    ):
        raise ValueError("Explicit roles and positive regularization required")
    rs, zs, targets, weights = [], [], [], []
    for p in refs:
        d = center_groups(p["displacement"], p["labels"])
        g = len(d) * center_groups(p["integrated"], p["labels"])
        rs.append((d @ base["vectors"]) ** 2)
        zs.append(token_conditions(p["body"], p["labels"], projection))
        targets.append(np.sum(d * g, axis=1))
        weights.append(np.full(len(d), 1 / (len(refs) * len(d))))
    r, z, target, w = map(np.concatenate, (rs, zs, targets, weights))
    mean = np.sum(w[:, None] * z, axis=0)
    std = np.sqrt(np.sum(w[:, None] * (z - mean) ** 2, axis=0))
    std = np.where(std > 1e-10, std, 1.0)
    z = (z - mean) / std
    model = {
        "vectors": base["vectors"],
        "base": base["action_refit"],
        "condition_projection": projection,
        "condition_mean": mean,
        "condition_std": std,
    }
    for name, cols in CONDITIONS.items():
        c = np.column_stack([np.ones(len(z)), z[:, cols]])
        x = (r[:, :, None] * c[:, None, :]).reshape(len(r), -1)
        scale = np.sqrt(np.sum(w[:, None] * x * x, axis=0))
        scale = np.where(scale > 1e-12, scale, 1.0)
        x /= scale
        prior = np.zeros((r.shape[1], c.shape[1]))
        prior[:, 0] = base["action_refit"]
        theta0 = prior.ravel() * scale
        gram = x.T @ (w[:, None] * x)
        rhs = x.T @ (w * target)
        theta = theta0 + np.linalg.solve(
            gram + alpha * np.eye(len(gram)), rhs - gram @ theta0
        )
        model[f"{name}_coef"] = (theta / scale).reshape(prior.shape)
        model[f"{name}_feature_scale"] = scale
    return model


def token_readouts(body, labels, model):
    """Apply frozen condition scales and fields; never refit on the scoring batch."""
    d = center_groups(body, labels)
    r = (d @ model["vectors"]) ** 2
    z = (
        token_conditions(body, labels, model["condition_projection"])
        - model["condition_mean"]
    ) / model["condition_std"]
    out = {"base": r @ model["base"]}
    for name, cols in CONDITIONS.items():
        c = np.column_stack([np.ones(len(z)), z[:, cols]])
        out[name] = np.sum(r * (c @ model[f"{name}_coef"].T), axis=1)
    return out
