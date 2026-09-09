#!/usr/bin/env python3
"""Small author-level reference and between-group response statistics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def checked_groups(labels, n):
    """Require matched nonempty groups rather than silently dropping a singleton."""
    labels = np.asarray(labels)
    if labels.shape != (n,) or len(set(labels)) != 2:
        raise ValueError("Two aligned author groups required")
    return labels


def separation_statistic(response, labels):
    """Compare within/between-author-group distances, with authors as the unit."""
    x = np.asarray(response, float)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) < 4:
        raise ValueError("Finite author-by-response matrix required")
    labels = checked_groups(labels, len(x))
    a, b = np.triu_indices(len(x), 1)
    distances = squareform(pdist(x, "sqeuclidean"))[a, b] / x.shape[1]
    within = labels[a] == labels[b]
    if not within.any() or within.all():
        raise ValueError("Both within and between group pairs required")
    gain = float(distances[~within].mean() - distances[within].mean())
    return {
        "within_mse": float(distances[within].mean()),
        "between_mse": float(distances[~within].mean()),
        "separation_gain": gain,
        "relative_gain": gain / max(float(distances.mean()), 1e-20),
        "within_pairs": int(within.sum()),
        "between_pairs": int((~within).sum()),
    }


def group_permutation(response, labels, draws=4999, seed=20260909):
    """Shuffle complete author group assignments, not dependent author pairs."""
    labels = checked_groups(labels, len(response))
    observed = separation_statistic(response, labels)
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            separation_statistic(response, rng.permutation(labels))["separation_gain"]
            for _ in range(draws)
        ]
    )
    return observed | {
        "permutation_p_greater": float(
            (1 + (null >= observed["separation_gain"]).sum()) / (draws + 1)
        ),
        "draws": draws,
        "null_mean": float(null.mean()),
    }


def distance_relation(profile, response, draws=4999, seed=20260909):
    """Associate cross-view distances using an author-index permutation null."""
    p, r = np.asarray(profile, float), np.asarray(response, float)
    if p.ndim != 2 or r.ndim != 2 or len(p) != len(r) or len(p) < 4:
        raise ValueError("Matched author profiles required")
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("Finite profiles required")
    a, b = np.triu_indices(len(p), 1)
    dx = squareform(pdist(p, "sqeuclidean"))[a, b]
    dy = squareform(pdist(r, "sqeuclidean"))
    observed = float(spearmanr(dx, dy[a, b]).statistic)
    if not np.isfinite(observed):
        return {"distance_spearman": None, "permutation_p_two_sided": None, "draws": 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        ix = rng.permutation(len(p))
        values.append(float(spearmanr(dx, dy[ix[a], ix[b]]).statistic))
    return {
        "distance_spearman": observed,
        "permutation_p_two_sided": float(
            (1 + (np.abs(values) >= abs(observed)).sum()) / (draws + 1)
        ),
        "draws": draws,
    }
