#!/usr/bin/env python3
"""Exact finite observation fibers and their irreducible squared-error bounds."""

from __future__ import annotations

import numpy as np


def collision_account(coordinates, responses):
    """Compute the best possible empirical map of coordinates to responses.

    This is a finite information bound, not fitted prediction performance.
    Equality is exact: no tolerance, clustering radius or target-based matching.
    """
    g, y = np.asarray(coordinates, float), np.asarray(responses, float)
    if g.ndim != 2 or y.ndim < 2 or len(g) != len(y) or not len(g):
        raise ValueError("Addressed finite observations and response rows required")
    if not np.isfinite(g).all() or not np.isfinite(y).all():
        raise ValueError("No missing observations may be silently filled")
    classes = {}
    for i, row in enumerate(g):
        classes.setdefault(tuple(row), []).append(i)
    means = np.empty_like(y)
    ids = np.empty(len(g), int)
    for index, members in enumerate(classes.values()):
        means[members] = y[members].mean(axis=0)
        ids[members] = index
    residual = y - means
    sizes = np.array([len(members) for members in classes.values()])
    return {
        "class_ids": ids,
        "class_sizes": sizes,
        "best_finite_mean": means,
        "irreducible_mse": np.mean(residual**2, axis=0),
        "collision_rows": int(sizes[sizes > 1].sum()),
        "distinct_observations": len(classes),
    }
