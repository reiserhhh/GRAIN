#!/usr/bin/env python3
"""Fixed-reference levels and qualified responses on the same author events."""

import numpy as np
from scipy.stats import rankdata

from grain_research.construction import math_prospective_evidence as P


def reference(bank):
    """Compile an equal-author/window SOURCE origin, never an application norm."""
    z, _, _ = P.validate(bank)
    return P.accounts(bank, z.reshape(-1, z.shape[-1])).mean(0)


def score(bank, vectors, event_ids, support, counter, origin):
    """Return levels and responses without outcome/teacher/model inputs."""
    z, _, _ = P.validate(bank)
    x = np.asarray(vectors, float)
    origin = np.asarray(origin, float)
    s, c = np.asarray(support), np.asarray(counter)
    if (
        x.shape != (12, z.shape[-1])
        or not np.isfinite(x).all()
        or len(event_ids) != 12
        or any(not isinstance(e, str) or not e for e in event_ids)
        or len(set(event_ids)) != 12
        or origin.shape != (3, 5)
        or not np.isfinite(origin).all()
        or abs(origin[0] - origin[1] - origin[2]).max() > 2e-11
        or s.shape != (4,)
        or c.shape != (4,)
        or s.dtype.kind not in "iu"
        or c.dtype.kind not in "iu"
        or len(set(s.tolist())) != 4
        or len(set(c.tolist())) != 4
        or set(s.tolist()) & set(c.tolist())
        or np.any((np.r_[s, c] < 0) | (np.r_[s, c] >= 12))
    ):
        raise ValueError(
            "Twelve distinct events, two disjoint four-event edits required"
        )
    points = np.stack(
        (x.mean(0), np.delete(x, s, 0).mean(0), np.delete(x, c, 0).mean(0))
    )
    values = P.accounts(bank, points)
    return {
        "points": points,
        "absolute": values,
        "positions": values - origin,
        "response": values[2] - values[1],
    }


def join(ids, other_ids, outcomes, other_outcomes, panel, source):
    """Fail closed on public numeric author/trait identity or source overlap."""
    ids, other_ids = np.asarray(ids), np.asarray(other_ids)
    y, same_y = np.asarray(outcomes, float), np.asarray(other_outcomes, float)
    panel, source = np.asarray(panel), np.asarray(source)
    if (
        ids.ndim != 1
        or len(set(ids.tolist())) != len(ids)
        or not np.array_equal(ids, other_ids)
        or y.shape != (len(ids), 5)
        or not np.array_equal(y, same_y)
        or not np.isfinite(y).all()
        or np.any((y < 0) | (y > 1))
        or panel.ndim != 1
        or source.ndim != 1
        or panel.dtype.kind not in "iu"
        or source.dtype.kind not in "iu"
        or len(set(panel.tolist())) != len(panel)
        or len(set(source.tolist())) != len(source)
        or set(panel.tolist()) & set(source.tolist())
        or np.any((np.r_[panel, source] < 0) | (np.r_[panel, source] >= len(ids)))
    ):
        raise ValueError("Public author identities/outcomes/roles do not match")
    return y[panel]


def covariance(x, y):
    """Return population-denominator cross covariance in declared input units."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Same whole-author rows required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite whole-author values required")
    return (x - x.mean(0)).T @ (y - y.mean(0)) / len(x)


def correlations(x, y):
    """Cross Pearson r, returning NaN rather than a fake zero for constants."""
    cov = covariance(x, y)
    nx = np.sqrt(np.diag(covariance(x, x)))
    ny = np.sqrt(np.diag(covariance(y, y)))
    denom = nx[:, None] * ny[None, :]
    return np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)


def associations(objects, y, draws, permutations, seed):
    """Use identical author resamples and joint five-outcome-row nulls."""
    x, y = np.asarray(objects, float), np.asarray(y, float)
    if x.ndim != 3 or x.shape[1:] != (10, 5) or y.shape != (len(x), 5):
        raise ValueError(
            "Ten fixed five-target objects and five linked outcomes required"
        )
    if draws < 1 or permutations < 1:
        raise ValueError("Positive fixed bootstrap/permutation counts required")
    flat = x.reshape(len(x), -1)
    observed = correlations(flat, y)
    rho = correlations(rankdata(flat, axis=0), rankdata(y, axis=0))
    rng = np.random.default_rng(seed)
    bootstrap_rows = rng.integers(len(x), size=(draws, len(x)))
    boot = np.stack([correlations(flat[ix], y[ix]) for ix in bootstrap_rows])
    permutation_rows = np.stack([rng.permutation(len(x)) for _ in range(permutations)])
    null = np.stack([correlations(flat, y[ix]) for ix in permutation_rows])
    point = (1 + (abs(null) >= abs(observed)).sum(0)) / (permutations + 1)
    max_null = np.nanmax(abs(null), axis=(1, 2))
    family = (1 + (max_null[:, None, None] >= abs(observed)).sum(0)) / (
        permutations + 1
    )
    point[~np.isfinite(observed)] = np.nan
    family[~np.isfinite(observed)] = np.nan
    return {
        "pearson": observed.reshape(10, 5, 5),
        "spearman": rho.reshape(10, 5, 5),
        "bootstrap": boot.reshape(draws, 10, 5, 5),
        "point_tail": point.reshape(10, 5, 5),
        "family_tail": family.reshape(10, 5, 5),
        "bootstrap_rows": bootstrap_rows,
        "permutation_rows": permutation_rows,
        "max_null": max_null,
    }


def objects(positions, responses, native):
    """Keep all level/response/source components and the native numerical mean."""
    pos, response, native = (
        np.asarray(a, float) for a in (positions, responses, native)
    )
    n = len(pos)
    if (
        pos.shape != (n, 3, 3, 5)
        or response.shape != (n, 3, 5)
        or native.shape != (n, 3, 5)
    ):
        raise ValueError(
            "All three same-author arms and complete score accounts required"
        )
    level, action = native[:, 0], native[:, 2] - native[:, 1]
    relative_level = level - level.mean(1, keepdims=True)
    relative_action = action - action.mean(1, keepdims=True)
    return np.stack(
        [pos[:, 0, j] for j in range(3)]
        + [response[:, j] for j in range(3)]
        + [level, action, relative_level, relative_action],
        axis=1,
    )
