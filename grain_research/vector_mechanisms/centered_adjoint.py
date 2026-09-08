#!/usr/bin/env python3
"""Finite conditional adjoint fields and outcome-blind low-rank quadratic readers."""

from __future__ import annotations

import numpy as np


def center_groups(values, labels):
    """Subtract each comment's vector mean, retaining all rows and coordinates."""
    x, groups = np.asarray(values, float), np.asarray(labels)
    if (
        x.ndim != 2
        or groups.shape != (len(x),)
        or not len(x)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Finite token-by-channel values and comment labels required")
    out = x.copy()
    for group in np.unique(groups):
        mask = groups == group
        out[mask] -= x[mask].mean(0)
    return out


def body_partition(state, groups):
    """Resolve disjoint ordered body rows and construct token/group coordinates."""
    h = np.asarray(state)
    if h.ndim != 2 or not groups or any(not len(g) for g in groups):
        raise ValueError("Token state and nonempty comment groups required")
    indices = np.concatenate([np.asarray(g, int) for g in groups])
    if (
        indices.min() < 0
        or indices.max() >= len(h) - 1
        or not np.array_equal(indices, np.unique(indices))
    ):
        raise ValueError(
            "Body groups must be ordered, disjoint and exclude final query"
        )
    labels = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    mask = np.zeros(len(h), bool)
    mask[indices] = True
    body = h[indices].astype(float)
    return mask, labels, body, center_groups(body, labels)


def integrate_nodes(gradients):
    """Integrate the frozen three-node or conditional five-node Simpson rule."""
    nodes = sorted(gradients)
    if nodes == [0.0, 0.5, 1.0]:
        weights = np.array([1, 4, 1]) / 6
    elif nodes == [0.0, 0.25, 0.5, 0.75, 1.0]:
        weights = np.array([1, 4, 2, 4, 1]) / 12
    else:
        raise ValueError("Three or five prescribed quadrature nodes required")
    arrays = [np.asarray(gradients[t], float) for t in nodes]
    if len({a.shape for a in arrays}) != 1 or not all(
        np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("Finite matched gradient matrices required")
    return sum(w * a for w, a in zip(weights, arrays, strict=True))


def field_account(displacement, gradient, labels):
    """Compare directional action, centered covariance and the rounding remainder."""
    d, g = np.asarray(displacement, float), np.asarray(gradient, float)
    if d.shape != g.shape:
        raise ValueError("Matched displacement and adjoint field required")
    dc, gc = center_groups(d, labels), center_groups(g, labels)
    action = float(np.sum(d * g))
    centered = float(np.sum(dc * gc))
    return {
        "action": action,
        "covariance_action": centered,
        "centering_remainder": action - centered,
        "constant_field_action": float(np.sum(dc * (g - gc))),
    }


def fit_reference_field(packets, rank=64, alpha=1.0, seed=20260908):
    """Fit only reference authors' normalized adjoints, with equal author mass."""
    refs = [p for p in packets if p["role"] == "reference"]
    if (
        not refs
        or alpha <= 0
        or any(p["role"] not in {"reference", "evaluation"} for p in packets)
    ):
        raise ValueError(
            "Explicit reference/evaluation roles and positive alpha required"
        )
    ds, gammas, weights = [], [], []
    for packet in refs:
        d = center_groups(packet["displacement"], packet["labels"])
        g = center_groups(packet["integrated"], packet["labels"])
        if d.shape != g.shape or not np.isfinite(g).all():
            raise ValueError("Finite reference gradients must match state coordinates")
        ds.append(d)
        gammas.append(len(d) * g)
        weights.append(np.full(len(d), 1 / (len(refs) * len(d))))
    d, gamma, w = np.concatenate(ds), np.concatenate(gammas), np.concatenate(weights)
    if not 1 <= rank <= d.shape[1]:
        raise ValueError("Projection rank must fit the hidden dimension")
    rng = np.random.default_rng(seed)
    projection = np.linalg.qr(rng.normal(size=(d.shape[1], rank)), mode="reduced")[0]
    x = d @ projection
    scale = np.sqrt(np.sum(w[:, None] * x**2, axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    x /= scale
    gram = x.T @ (w[:, None] * x)
    cross = x.T @ (w[:, None] * gamma)
    coef = np.linalg.solve(gram + alpha * np.eye(rank), cross)
    energy = float(np.sum(w[:, None] * d**2))
    radial = float(np.sum(w[:, None] * d * gamma) / energy) if energy else 0.0
    return {
        "projection": projection,
        "scale": scale,
        "coef": coef,
        "radial": np.asarray(radial),
    }


def read_field(displacement, labels, model):
    """Evaluate fixed anisotropic/radial fields without any model suffix or gradient."""
    d = center_groups(displacement, labels)
    p, scale, coef = (
        np.asarray(model[k], float) for k in ("projection", "scale", "coef")
    )
    if p.shape != (d.shape[1], len(scale)) or coef.shape != (len(scale), d.shape[1]):
        raise ValueError("Frozen field has incompatible hidden coordinates")
    x = d @ p / scale
    return {
        "rank64_field": float(np.sum(x * (d @ coef.T)) / len(d)),
        "isotropic_field": float(model["radial"] * np.sum(d**2) / len(d)),
        "constant_field": 0.0,
    }
