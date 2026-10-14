#!/usr/bin/env python3
"""Exact support-aware decomposition of observed semantic-profile differences."""

from __future__ import annotations

import numpy as np

PARTS = ("total", "within", "composition", "turnover")


def chronological_halves(timestamps, comment_ids):
    """Split an even panel by actual time; separately flag ties at its boundary."""
    ts = np.asarray(timestamps, float)
    ids = np.asarray(comment_ids, str)
    if ts.ndim != 1 or len(ts) < 2 or len(ts) % 2 or len(ids) != len(ts):
        raise ValueError("An aligned even comment panel is required")
    if not np.isfinite(ts).all() or len(set(ids)) != len(ids):
        raise ValueError("Finite timestamps and unique comment IDs required")
    order = np.lexsort((ids, ts))
    n = len(ts) // 2
    half = np.zeros(len(ts), int)
    half[order[n:]] = 1
    return half, {
        "strict_time_boundary": bool(ts[order[n - 1]] < ts[order[n]]),
        "span_days": float((ts.max() - ts.min()) / 86400),
        "boundary_gap_days": float((ts[order[n]] - ts[order[n - 1]]) / 86400),
    }


def comment_average(values, comment_indices, count):
    """Aggregate candidate scores by comment, retaining absent comments as NaN."""
    values = np.asarray(values, float)
    ix = np.asarray(comment_indices)
    if values.ndim != 2 or len(values) != len(ix) or not np.isfinite(values).all():
        raise ValueError("Aligned finite sentence scores required")
    if not np.issubdtype(ix.dtype, np.integer) or np.any(ix < 0) or np.any(ix >= count):
        raise ValueError("Invalid comment address")
    counts = np.bincount(ix, minlength=count)
    sums = np.zeros((count, values.shape[1]))
    np.add.at(sums, ix, values)
    means = np.full_like(sums, np.nan)
    np.divide(sums, counts[:, None], out=means, where=counts[:, None] > 0)
    return means, counts


def difference_operator(conditions, halves, available):
    """Build total/within/composition/turnover weights using metadata only.

    The rows of the returned matrix act on comments. Columns without an
    observed score have zero weight. A zero within component without shared
    support is bookkeeping, not a measured absence of change.
    """
    c, h, observed = (
        np.asarray(conditions, str),
        np.asarray(halves),
        np.asarray(available),
    )
    if c.ndim != 1 or h.shape != c.shape or observed.shape != c.shape:
        raise ValueError("Metadata dimensions differ")
    if observed.dtype != bool or not np.isin(h, [0, 1]).all():
        raise ValueError("Boolean support and binary halves required")
    ix = [np.flatnonzero(observed & (h == t)) for t in (0, 1)]
    if min(map(len, ix)) == 0:
        raise ValueError("Both halves require observed comment scores")
    sets = [set(c[j]) for j in ix]
    shared = sets[0] & sets[1]
    a = np.zeros((4, len(c)))
    a[0, ix[0]], a[0, ix[1]] = -1 / len(ix[0]), 1 / len(ix[1])
    mass = [0.0, 0.0]
    repeated, shared_comments = 0, [0, 0]
    for key in sorted(sets[0] | sets[1]):
        cells = [j[c[j] == key] for j in ix]
        n0, n1 = map(len, cells)
        p0, p1 = n0 / len(ix[0]), n1 / len(ix[1])
        if key in shared:
            a[1, cells[0]] = -(p0 + p1) / (2 * n0)
            a[1, cells[1]] = (p0 + p1) / (2 * n1)
            a[2, cells[0]] = (p1 - p0) / (2 * n0)
            a[2, cells[1]] = (p1 - p0) / (2 * n1)
            mass[0] += p0
            mass[1] += p1
            shared_comments[0] += n0
            shared_comments[1] += n1
            repeated += min(n0, n1) >= 2
        else:
            for t, j in enumerate(cells):
                a[3, j] = (-1 if t == 0 else 1) / len(ix[t])
    if not np.allclose(a[0], a[1:].sum(0), atol=1e-13):
        raise ArithmeticError("Difference weights do not add up")
    return a, {
        "early_comments": len(ix[0]),
        "late_comments": len(ix[1]),
        "shared_conditions": len(shared),
        "repeated_shared_conditions": int(repeated),
        "early_shared_mass": mass[0],
        "late_shared_mass": mass[1],
        "early_shared_comments": shared_comments[0],
        "late_shared_comments": shared_comments[1],
        "within_measured": bool(shared),
    }


def apply_operator(operator, comment_scores):
    """Apply only to supported observations; never impute missing-cell responses."""
    a, y = np.asarray(operator, float), np.asarray(comment_scores, float)
    if a.ndim != 2 or y.ndim != 2 or a.shape[1] != len(y) or not np.isfinite(a).all():
        raise ValueError("Aligned finite operator required")
    used = np.any(a != 0, axis=0)
    if not np.isfinite(y[used]).all():
        raise ValueError("Operator attempted to use missing scores")
    return a[:, used] @ y[used]


def direct_difference(conditions, halves, scores):
    """Independent cell-mean product calculation, used for arithmetic auditing."""
    c, h, y = np.asarray(conditions, str), np.asarray(halves), np.asarray(scores, float)
    valid = np.isfinite(y).all(1)
    cells = []
    means = []
    for t in (0, 1):
        sel = (h == t) & valid
        if not sel.any():
            raise ValueError("Missing half")
        cells.append(
            {
                key: (np.mean(c[sel] == key), y[sel & (c == key)].mean(0))
                for key in set(c[sel])
            }
        )
        means.append(y[sel].mean(0))
    output = np.zeros((4, y.shape[1]))
    output[0] = means[1] - means[0]
    for key in set(cells[0]) | set(cells[1]):
        if key in cells[0] and key in cells[1]:
            p0, m0 = cells[0][key]
            p1, m1 = cells[1][key]
            output[1] += (p0 + p1) / 2 * (m1 - m0)
            output[2] += (p1 - p0) * (m1 + m0) / 2
        elif key in cells[0]:
            output[3] -= cells[0][key][0] * cells[0][key][1]
        else:
            output[3] += cells[1][key][0] * cells[1][key][1]
    return output


def paired_energy_interval(chronological, random_mean, draws=500, seed=914):
    """Author bootstrap of a descriptive time-vs-random squared-change contrast."""
    d = np.asarray(chronological, float) - np.asarray(random_mean, float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError("One finite paired energy per author required")
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(len(d), size=(draws, len(d)))].mean(1)
    return {
        "delta_mean_energy": float(d.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "authors": len(d),
        "draws": draws,
    }
