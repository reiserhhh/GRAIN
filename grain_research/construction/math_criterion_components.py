#!/usr/bin/env python3
"""Separate retained criterion composition from conditional paired readings."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_author_pair_criterion as A
from grain_research.construction import math_source_anchor_exchange as E
from grain_research.construction import math_source_event_coupling as P

CHANNELS = ("total", "composition", "within_configuration")


def criterion_design(labels, banks):
    """Compile criterion means/departures in each bank, requiring all32 profiles."""
    y, b = np.asarray(labels, float), np.asarray(banks)
    if (
        y.ndim != 2
        or y.shape[1] != 5
        or b.shape != (len(y),)
        or set(np.unique(b)) != {0, 1}
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError("Two banks and complete five binary criteria required")
    means = np.empty_like(y)
    groups = np.empty(y.shape, int)
    for bank in (0, 1):
        ix = np.flatnonzero(b == bank)
        for q in range(5):
            z, counts, _, _ = E.design(y[ix], q)
            means[ix, q] = (counts[1] / counts.sum(0))[z]
            groups[ix, q] = z + 16 * bank
    return means, y - means, groups


def score_components(weights, labels, banks):
    """Retain all three source-only channels, without an application criterion."""
    d = np.asarray(weights, float)
    y = np.asarray(labels, float)
    b = np.asarray(banks)
    if (
        d.ndim != 4
        or d.shape[1:] != (len(y), 3, len(y))
        or not np.isfinite(d).all()
        or np.any(d[:, :, :, :].transpose(0, 2, 1, 3)[:, :, b[:, None] == b] != 0)
    ):
        raise ValueError("Finite cross-bank space/author/three-window weights required")
    means, departure, groups = criterion_design(y, b)
    scores = np.stack(
        [np.einsum("suwj,jq->suwq", d, v) for v in (y, means, departure)], 3
    )
    return scores, means, departure, groups


def group_center(values, groups):
    """Center a statistic within its declared groups, not text representations."""
    x = np.asarray(values, float).copy()
    g = np.asarray(groups)
    if x.shape != g.shape or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite author/target values and matched group IDs required")
    for q in range(x.shape[1]):
        for cell in np.unique(g[:, q]):
            ix = g[:, q] == cell
            x[ix, q] -= x[ix, q].mean()
    return x


def conditional_r(scores, departure, groups):
    """Report within-other-four correlations, not all conditional independence."""
    x = group_center(scores, groups)
    y = group_center(departure, groups)
    denominator = np.sqrt(np.sum(x * x, 0) * np.sum(y * y, 0))
    return np.divide(
        np.sum(x * y, 0),
        denominator,
        out=np.full(x.shape[1], np.nan),
        where=denominator > 0,
    )


def orbit_covariance(labels, groups):
    """Exact centered-label covariance under within-group finite permutations."""
    y, g = np.asarray(labels, float), np.asarray(groups)
    if y.ndim != 1 or g.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("A binary target and one group ID per author required")
    v = np.zeros((len(y), len(y)))
    for cell in np.unique(g):
        ix = np.flatnonzero(g == cell)
        m, p = len(ix), y[ix].mean()
        if m > 1:
            v[np.ix_(ix, ix)] = (
                p * (1 - p) * m / (m - 1) * (np.eye(m) - np.ones((m, m)) / m)
            )
    return v


def orbit_variance(weights, labels, banks, target, groups):
    """Include both cross-fit directions in the exact conditional-orbit variance."""
    d, y, b = np.asarray(weights, float), np.asarray(labels), np.asarray(banks)
    ix, jx = np.flatnonzero(b == 0), np.flatnonzero(b == 1)
    h = d[np.ix_(ix, jx)] + d[np.ix_(jx, ix)].T
    v0 = orbit_covariance(y[ix, target], groups[ix, target])
    v1 = orbit_covariance(y[jx, target], groups[jx, target])
    return float(np.sum((v0 @ h @ v1) * h) / len(y) ** 2)


def assignment_reference(weights, labels, banks, count, seed):
    """Reassign SOURCE and held target together, preserving all other-four rows."""
    d, y, b = np.asarray(weights, float), np.asarray(labels, float), np.asarray(banks)
    _, departure, groups = criterion_design(y, b)
    ix, jx = np.flatnonzero(b == 0), np.flatnonzero(b == 1)
    if d.ndim != 3 or d.shape[1:] != (len(y), len(y)):
        raise ValueError("Space-by-author opposite-bank weight matrices required")
    if np.any(d[:, b[:, None] == b[None, :]] != 0):
        raise ValueError("Own-bank source weights must be zero")
    observed = np.einsum("uq,suj,jq->sq", departure, d, departure) / len(y)
    null = np.empty((count, len(d), 5))
    assignments = np.empty((5, count, len(y)), int)
    variances = np.empty_like(observed)
    h = d[:, ix][:, :, jx] + d[:, jx][:, :, ix].transpose(0, 2, 1)
    for q in range(5):
        p0 = E.assignment_indices(y[ix], q, count, seed)
        p1 = E.assignment_indices(y[jx], q, count, seed + 1)
        assignments[q, :, ix] = ix[p0].T
        assignments[q, :, jx] = jx[p1].T
        e0, e1 = departure[ix, q][p0], departure[jx, q][p1]
        null[:, :, q] = np.einsum("ki,sij,kj->ks", e0, h, e1) / len(y)
        for sp in range(len(d)):
            variances[sp, q] = orbit_variance(d[sp], y, b, q, groups)
    nominal = (1 + np.sum(abs(null) >= abs(observed), axis=0)) / (count + 1)
    holm = np.asarray(E.holm(nominal.ravel())).reshape(nominal.shape)
    return observed, null, variances, nominal, holm, assignments


def cold_components(bundle, pairs, space):
    """Read one new author's numeric pairs from SOURCE criteria only."""
    x, z = np.asarray(pairs, float), np.asarray(bundle["prototypes"], float)
    if x.shape != (22, 2, z.shape[-1]) or space not in ("E5", "GRAIN15"):
        raise ValueError("Registered22 numeric pairs and frozen space required")
    if space == "GRAIN15":
        x, z = x @ bundle["projection"], z @ bundle["projection"]
    labels = P.BITS[bundle["codes"]]
    means = np.empty_like(labels)
    for q in range(5):
        g, counts, _, _ = E.design(labels, q)
        means[:, q] = (counts[1] / counts.sum(0))[g]
    w = A.source_author_weights(
        P.event_logs(x, z, float(bundle["variance"])), bundle["relink"]
    )
    d = (w[:, 0] - w[:, 1]).reshape(2, 11, len(z)).mean(1)
    d = np.concatenate((d, d.mean(0, keepdims=True)))
    return np.stack([d @ v for v in (labels, means, labels - means)], 1)
