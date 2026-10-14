#!/usr/bin/env python3
"""Finite unequal-event geometry, retaining all within-condition variation."""

import numpy as np

from . import math_condition_pairing as P

VERSION = "HM189_distinct_condition_geometry_v1"
GEOMETRIES = ("squared", "distinct_event", "within_condition_term")


def observations(vectors, conditions):
    """Separate self-products from unequal-event products, not psychological noise."""
    x, c = np.asarray(vectors, float), np.asarray(conditions)
    first = P.moments(x, c)
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError("Two addressed author views required")
    counts = np.stack(((c == 0).sum(-1), (c == 1).sum(-1)), -1)
    support = (counts >= 2).all(-1)
    g = np.full((*x.shape[:2], 3, x.shape[-1], x.shape[-1]), np.nan)
    for u, v in zip(*np.where(support), strict=True):
        raw = np.outer(first["delta"][u, v], first["delta"][u, v])
        term = np.zeros_like(raw)
        for side in (0, 1):
            row = x[u, v, c[u, v] == side]
            centered = row - row.mean(0)
            term += centered.T @ centered / (len(row) * (len(row) - 1))
        g[u, v] = np.stack((raw, raw - term, term))
    return {
        "a": (1 - 2 * first["p"]) * first["opportunity"],
        "support": support,
        "counts": counts,
        "geometry": g,
        "conditional_support": first["support"],
    }


def literal_observations(vectors, conditions):
    """Construct unequal-event products directly, without covariance subtraction."""
    x, c = np.asarray(vectors, float), np.asarray(conditions)
    counts = np.stack(((c == 0).sum(-1), (c == 1).sum(-1)), -1)
    s = (counts >= 2).all(-1)
    g = np.full((*x.shape[:2], 3, x.shape[-1], x.shape[-1]), np.nan)
    for u, v in zip(*np.where(s), strict=True):
        row0, row1 = (x[u, v, c[u, v] == side] for side in (0, 1))
        mu0, mu1 = row0.mean(0), row1.mean(0)
        same = []
        for row in (row0, row1):
            total = row.sum(0)
            same.append(
                (np.outer(total, total) - row.T @ row) / (len(row) * (len(row) - 1))
            )
        U = same[0] + same[1] - np.outer(mu0, mu1) - np.outer(mu1, mu0)
        raw = np.outer(mu1 - mu0, mu1 - mu0)
        g[u, v] = np.stack((raw, U, raw - U))
    p = c.mean(-1)
    return {
        "a": (1 - 2 * p) * p * (1 - p),
        "support": s,
        "counts": counts,
        "geometry": g,
        "conditional_support": (counts > 0).all(-1),
    }
