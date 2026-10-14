#!/usr/bin/env python3
"""Exact finite observation fibers, empirical information loss and support."""

from __future__ import annotations

import itertools

import numpy as np

WORLDS = (
    "complete_linear",
    "omitted_coordinate",
    "shared_nonlinear",
    "injective_support",
)


def finite_matrix(values):
    """Require finite nonempty matrices; no approximate equality or rounding."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Nonempty finite numeric matrix required")
    return x


def exact_partition(values):
    """Partition by exact numeric equality, including equivalent signed zeros."""
    x = finite_matrix(values)
    keys, inverse, counts = np.unique(
        x, axis=0, return_inverse=True, return_counts=True
    )
    return keys, inverse, counts


def author_weights(authors):
    """Equal author mass, then equal comment mass inside each author."""
    a = np.asarray(authors)
    if a.ndim != 1 or not len(a):
        raise ValueError("Nonempty aligned author IDs required")
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    return 1 / (len(counts) * counts[inverse])


def empirical_projection(inputs, targets, weights):
    """Optimal finite class-constant output, not a transferable learned law."""
    x, y, w = finite_matrix(inputs), finite_matrix(targets), np.asarray(weights, float)
    if len(x) != len(y) or w.shape != (len(x),) or not np.isfinite(w).all():
        raise ValueError("Aligned finite observations and weights required")
    if np.any(w <= 0) or abs(w.sum() - 1) > 1e-10:
        raise ValueError("Strictly positive probability weights required")
    keys, inverse, counts = exact_partition(x)
    mass = np.bincount(inverse, weights=w, minlength=len(keys))
    sums = np.zeros((len(keys), y.shape[1]))
    np.add.at(sums, inverse, w[:, None] * y)
    means = sums / mass[:, None]
    residual = y - means[inverse]
    risk = np.sum(w[:, None] * residual**2, axis=0)
    center = np.sum(w[:, None] * y, axis=0)
    total = np.sum(w[:, None] * (y - center) ** 2, axis=0)
    return {
        "keys": keys,
        "inverse": inverse,
        "counts": counts,
        "mass": mass,
        "means": means,
        "residual": residual,
        "risk": risk,
        "total": total,
        "fraction": np.divide(risk, total, out=np.zeros_like(risk), where=total > 0),
    }


def lookup(projection, inputs):
    """Transport exact source-class means only; unseen inputs stay unobserved."""
    x = finite_matrix(inputs)
    if x.shape[1] != projection["keys"].shape[1]:
        raise ValueError("Same input coordinate width required")
    mapping = {tuple(key): i for i, key in enumerate(projection["keys"])}
    indexes = np.array([mapping.get(tuple(row), -1) for row in x], dtype=int)
    seen = indexes >= 0
    result = np.full((len(x), projection["means"].shape[1]), np.nan)
    result[seen] = projection["means"][indexes[seen]]
    return result, seen


def roundoff_envelope(embedding, weights):
    """Conservative fixed float64 dot envelope, never an input similarity rule."""
    h, w = finite_matrix(embedding), finite_matrix(weights)
    if h.shape[1] != w.shape[0]:
        raise ValueError("Aligned embedding and query weights required")
    de = h.shape[1] * np.finfo(float).eps
    if de >= 1:
        raise ValueError("Dimension outside finite roundoff bound")
    return 8 * de / (1 - de) * (abs(h) @ abs(w))


def fiber_extrema(inputs, targets, errors=None):
    """Retain every repeated-class/query extremum, including constant controls."""
    x, y = finite_matrix(inputs), finite_matrix(targets)
    e = np.zeros_like(y) if errors is None else np.asarray(errors, float)
    if (
        len(x) != len(y)
        or e.shape != y.shape
        or not np.isfinite(e).all()
        or (e < 0).any()
    ):
        raise ValueError("Aligned finite targets and nonnegative envelopes required")
    _, inverse, counts = exact_partition(x)
    rows = []
    for c in np.flatnonzero(counts > 1):
        indexes = np.flatnonzero(inverse == c)
        for q in range(y.shape[1]):
            lo, hi = (
                indexes[np.argmin(y[indexes, q])],
                indexes[np.argmax(y[indexes, q])],
            )
            gap = float(y[hi, q] - y[lo, q])
            bound = float(e[lo, q] + e[hi, q])
            rows.append(
                {
                    "class": int(c),
                    "query_index": q,
                    "low": int(lo),
                    "high": int(hi),
                    "size": int(counts[c]),
                    "gap": gap,
                    "roundoff_bound": bound,
                    "raw_nonconstant": gap > 0,
                    "certified": gap > bound,
                    "worst_absolute_error_bound": gap / 2,
                    "two_distribution_mean_gap": gap / 2,
                }
            )
    return rows


def synthetic_world(world, seed):
    """Actual numeric worlds distinguish missing inputs and finite lookup."""
    if world not in WORLDS:
        raise ValueError("Unknown mechanism world")
    rng = np.random.default_rng(seed)
    grid = np.array(list(itertools.product((-1.0, 0.0, 2.0), repeat=4)))
    base = grid[rng.permutation(len(grid))[:8]]
    gs, hs, authors, panels = [], [], [], []
    for author, panel in itertools.product(range(40), range(3)):
        g = base[rng.permutation(8)].copy()
        hidden = np.sin(author / 5) + panel / 3 + np.arange(8) / 8
        if world == "injective_support":
            g[:, 0] += (author * 24 + panel * 8 + np.arange(8) + 1) / 65536
        h = (
            np.column_stack((g, hidden))
            if world in ("omitted_coordinate", "injective_support")
            else g**2
            if world == "shared_nonlinear"
            else g.copy()
        )
        gs.append(g)
        hs.append(h)
        authors.extend([author] * 8)
        panels.extend([panel] * 8)
    h = np.vstack(hs)
    w = rng.normal(size=(h.shape[1], 3))
    if h.shape[1] == 5:
        w[-1] = (1, -1, 2)
    return {
        "g": np.vstack(gs),
        "h": h,
        "W": w,
        "y": h @ w,
        "author": np.array(authors),
        "panel": np.array(panels),
        "source": np.array(authors) < 20,
    }
