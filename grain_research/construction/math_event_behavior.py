#!/usr/bin/env python3
"""Distinguish author-position associations from identical-event secant contacts."""

import numpy as np


def support_transfer(values, actual_field, other_field, operation):
    """Transport only identified finite-alphabet actions; retain all multiplicities."""
    v, w, t = (np.asarray(a, float) for a in (values, actual_field, other_field))
    if (
        v.ndim != 2
        or w.shape != (8, 5)
        or t.shape != w.shape
        or v.shape[1] != 5
        or not all(np.isfinite(a).all() for a in (v, w, t))
        or operation not in ("difference", "mean")
    ):
        raise ValueError(
            "Finite declared eight-cell fields and identified operation required"
        )
    pairs = [(i, j) for i in range(8) for j in range(8)]
    combine = (
        (lambda a, b: a - b) if operation == "difference" else lambda a, b: (a + b) / 2
    )
    source = np.array([combine(w[i], w[j]) for i, j in pairs])
    target = np.array([combine(t[i], t[j]) for i, j in pairs])
    result, counts = [], []
    for row in v:
        ix = np.flatnonzero(np.max(abs(source - row), axis=1) <= 1e-12)
        if not len(ix):
            raise ValueError(
                "No matching registered cell action; do not infer unknown input"
            )
        if np.max(abs(target[ix] - target[ix[0]])) > 1e-12:
            raise ValueError(
                "Transformed action is not identified; do not average ambiguities"
            )
        result.append(target[ix[0]])
        counts.append(len(ix))
    return np.array(result), counts


def contacts(source_change, behavior_change):
    """Keep uncentered action, centered covariance and mean-product terms exactly."""
    x, y = (np.asarray(a, float) for a in (source_change, behavior_change))
    if x.ndim != 2 or y.shape != x.shape or x.shape[1] != 5 or not len(x):
        raise ValueError("Same addressed rows and five targets required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Missing observations cannot be treated as zero")
    centered = (x - x.mean(0)).T @ (y - y.mean(0)) / len(x)
    means = np.outer(x.mean(0), y.mean(0))
    return {"action": x.T @ y / len(x), "covariance": centered, "means": means}


def paired_intervals(source_change, behavior_change, seed, draws):
    """Resample authors with both dependent directions; do not count them as people."""
    x, y = (np.asarray(a, float) for a in (source_change, behavior_change))
    if x.shape != y.shape or x.ndim != 3 or x.shape[0] != 2 or x.shape[2] != 5:
        raise ValueError("Two directions x authors x five targets required")
    if draws < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite complete paired observations required")
    products = (x * y).mean(0)
    rng = np.random.default_rng(seed)
    sampled = products[rng.integers(0, len(products), (draws, len(products)))].mean(1)
    return np.quantile(sampled, [0.025, 0.975], axis=0).T
