#!/usr/bin/env python3
"""Signed cross-cohort configuration operators, without teacher-score fitting.

The Lancaster embedding is a third central tensor in an RKHS, not proof of
information absent from every pair distribution or of psychological meaning.
"""

from __future__ import annotations

import numpy as np


def finite_matrix(value):
    """Require a nonempty, finite two-dimensional numerical observation."""
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("nonempty finite matrix required")
    return x


def squared_distance(x, y, precision=None):
    """Compute metric distances; transport the metric when changing coordinates."""
    x, y = finite_matrix(x), finite_matrix(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("observation dimensions differ")
    xp, yp = x, y
    if precision is not None:
        p = finite_matrix(precision)
        if p.shape != (x.shape[1], x.shape[1]) or not np.allclose(p, p.T):
            raise ValueError("symmetric metric of the observation dimension required")
        if np.linalg.eigvalsh(p).min() <= 0:
            raise ValueError("positive definite metric required")
        xp, yp = x @ p, y @ p
    d = (xp * x).sum(1)[:, None] + (yp * y).sum(1)[None, :] - 2 * xp @ y.T
    if d.min() < -1e-7:
        raise ValueError("numerically invalid squared distances")
    return np.maximum(d, 0)


def source_bandwidth(source):
    """Set one scale from source pair distances, never from application scores."""
    d = squared_distance(source, source)
    upper = d[np.triu_indices(len(d), 1)]
    positive = upper[upper > 1e-14]
    if not len(positive):
        raise ValueError("source observations do not identify a positive kernel scale")
    return float(np.median(positive))


def rbf(x, y, bandwidth_squared, precision=None):
    """Return a Gaussian Gram matrix in the frozen observation metric."""
    if not np.isfinite(bandwidth_squared) or bandwidth_squared <= 0:
        raise ValueError("positive frozen bandwidth required")
    return np.exp(-squared_distance(x, y, precision) / (2 * bandwidth_squared))


def centered(matrix, left_weights=None, right_weights=None):
    """Center rectangular kernels at two empirical populations, retaining signs."""
    a = finite_matrix(matrix)
    w = weights(left_weights, a.shape[0])
    v = weights(right_weights, a.shape[1])
    return a - (a @ v)[:, None] - (w @ a)[None, :] + float(w @ a @ v)


def weights(value, n):
    """Validate population weights; zero-weight rows remain legal bootstrap rows."""
    w = np.full(n, 1 / n) if value is None else np.asarray(value, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or w.min() < 0:
        raise ValueError("nonnegative finite population weights required")
    if not np.isclose(w.sum(), 1):
        raise ValueError("population weights must sum to one")
    return w


def interaction_inner(matrices, left_weights=None, right_weights=None):
    """Contract signed second/third interaction embeddings of two populations."""
    mats = [finite_matrix(m) for m in matrices]
    if len(mats) not in (2, 3) or len({m.shape for m in mats}) != 1:
        raise ValueError("two or three aligned kernel matrices required")
    w, v = (
        weights(left_weights, mats[0].shape[0]),
        weights(right_weights, mats[0].shape[1]),
    )
    product = np.ones_like(mats[0])
    for m in mats:
        product *= centered(m, w, v)
    return float(w @ product @ v)


def total_independence(grams):
    """Compute dHSIC for three observations; it includes pair dependencies."""
    mats = [finite_matrix(g) for g in grams]
    if len(mats) != 3 or len({m.shape for m in mats}) != 1:
        raise ValueError("three aligned square Gram matrices required")
    if mats[0].shape[0] != mats[0].shape[1]:
        raise ValueError("square Gram matrices required")
    joint = np.prod(mats, axis=0).mean()
    marginal = float(np.prod([m.mean() for m in mats]))
    mixed = np.prod([m.mean(axis=1) for m in mats], axis=0).mean()
    return float(joint + marginal - 2 * mixed)


def source_witness(source_grams, cross_grams):
    """Score cold triples with a source-only signed tensor witness.

    Application means do not enter these per-triple readings. They are not
    calibrated personal scores; the cross-cohort interaction statistic separately
    centers both populations to describe their configuration laws.
    """
    if len(source_grams) != 3 or len(cross_grams) != 3:
        raise ValueError("three source and three cross kernels required")
    product = np.ones_like(finite_matrix(cross_grams[0]))
    for src, cross in zip(source_grams, cross_grams, strict=True):
        s, c = finite_matrix(src), finite_matrix(cross)
        if s.shape != (c.shape[0], c.shape[0]) or c.shape != product.shape:
            raise ValueError("source and application kernel addresses differ")
        product *= c - c.mean(0)[None, :] - s.mean(1)[:, None] + s.mean()
    return product.mean(0)


def permutation(n, rng, groups=None):
    """Relink author addresses globally or within predeclared coarse conditions."""
    result = np.arange(n)
    if groups is None:
        return rng.permutation(result)
    g = np.asarray(groups)
    if g.shape != (n,):
        raise ValueError("one condition stratum per author required")
    for label in np.unique(g):
        indices = np.flatnonzero(g == label)
        result[indices] = rng.permutation(indices)
    return result


def cut_readings(cross, permutations):
    """Evaluate all three single-view cuts, preserving the other pair exactly."""
    c = [centered(m) for m in cross]
    if len(c) != 3 or len({m.shape for m in c}) != 1:
        raise ValueError("three aligned cross kernels required")
    p = np.asarray(permutations)
    n = c[0].shape[1]
    if p.ndim != 2 or p.shape[1] != n:
        raise ValueError("aligned permutation addresses required")
    if not np.all(np.sort(p, axis=1) == np.arange(n)):
        raise ValueError("each relink must be a permutation, not a bootstrap")
    out = np.empty((len(p), 3))
    for j in range(3):
        other = c[(j + 1) % 3] * c[(j + 2) % 3]
        for i, order in enumerate(p):
            out[i, j] = (other * c[j][:, order]).mean()
    return out


def bootstrap_inner(cross, draws, seed, resample_source=True):
    """Resample complete authors, conditional on frozen features and kernel units."""
    rng = np.random.default_rng(seed)
    n, m = finite_matrix(cross[0]).shape
    values = []
    for _ in range(draws):
        w = rng.multinomial(n, np.full(n, 1 / n)) / n if resample_source else None
        v = rng.multinomial(m, np.full(m, 1 / m)) / m
        values.append(interaction_inner(cross, w, v))
    return np.asarray(values)


def balanced_views(venues, times):
    """Partition all event addresses into three venue-balanced, disjoint parcels."""
    venue, times = np.asarray(venues), np.asarray(times, dtype=float)
    if venue.ndim != 1 or times.shape != venue.shape or len(venue) % 3:
        raise ValueError("event counts must divide into three nonempty parcels")
    if not len(venue) or not np.isfinite(times).all():
        raise ValueError("finite event times required")
    labels, counts = np.unique(venue, return_counts=True)
    order = sorted(range(len(labels)), key=lambda j: (-counts[j], str(labels[j])))
    allocated = [[] for _ in range(3)]
    for j in order:
        indices = np.flatnonzero(venue == labels[j])
        indices = indices[np.argsort(times[indices], kind="stable")]
        for index in indices:
            arm = min(range(3), key=lambda k: (len(allocated[k]), k))
            allocated[arm].append(int(index))
    return np.asarray([sorted(view, key=lambda i: times[i]) for view in allocated])
