#!/usr/bin/env python3
"""Exact complementary-view laws conditional on a fixed event bank."""

from __future__ import annotations

from itertools import permutations, product

import numpy as np

PERMUTATIONS = np.array(list(permutations(range(3))), dtype=int)
METHODS = ("orbit_proxy", "finite_bank", "single_view")


def event_features(coordinates, conditions, units):
    """Keep all events, condition choice and both conditional vector masses."""
    z, c, ids = map(np.asarray, (coordinates, conditions, units))
    if (
        z.ndim != 3
        or z.shape[1] % 3
        or c.shape != z.shape[:2]
        or ids.shape != c.shape
        or not np.isfinite(z).all()
        or not np.isin(c, [0, 1]).all()
        or any(len(np.unique(row)) != len(row) for row in ids)
    ):
        raise ValueError("Distinct finite events and binary conditions required")
    return np.concatenate(
        [
            np.ones((*c.shape, 1)),
            c[..., None],
            (1 - c[..., None]) * z,
            c[..., None] * z,
        ],
        axis=-1,
    )


def orbit_moments(q):
    """Return the fixed-bank mean and negative cross-view covariance magnitude."""
    q = np.asarray(q, float)
    if q.ndim != 3 or q.shape[1] % 3 or not q.shape[1] or not np.isfinite(q).all():
        raise ValueError("Nonempty finite consecutive triples required")
    blocks = q.reshape(len(q), -1, 3, q.shape[-1])
    centered = blocks - blocks.mean(axis=2, keepdims=True)
    gamma = np.einsum("ubjd,ubje->ude", centered, centered) / (6 * blocks.shape[1] ** 2)
    return q.mean(axis=1), gamma


def partition_views(q, draws, seed):
    """Sample one of all six assignments independently in each unchanged triple."""
    q = np.asarray(q, float)
    orbit_moments(q)
    if draws < 1:
        raise ValueError("Positive fixed draw count required")
    blocks = q.reshape(len(q), -1, 3, q.shape[-1])
    rng = np.random.default_rng(seed)
    assignments = PERMUTATIONS[rng.integers(0, 6, (len(q), draws, blocks.shape[1]))]
    picked = blocks[
        np.arange(len(q))[:, None, None, None],
        np.arange(blocks.shape[1])[None, None, :, None],
        assignments,
    ]
    return picked.mean(axis=2), assignments


def construct(mu, gamma, y, tolerance=1e-12):
    """Construct three SOURCE moment operators; never select by held outcomes."""
    mu, gamma, y = (np.asarray(a, float) for a in (mu, gamma, y))
    if (
        mu.ndim != 2
        or gamma.shape != (len(mu), mu.shape[1], mu.shape[1])
        or y.ndim != 2
        or len(y) != len(mu)
        or not all(np.isfinite(a).all() for a in (mu, gamma, y))
    ):
        raise ValueError(
            "Aligned finite source means, covariance and criteria required"
        )
    mtx, g, rhs = mu.T @ mu / len(mu), gamma.mean(axis=0), mu.T @ y / len(mu)
    matrices = {"orbit_proxy": mtx - g, "finite_bank": mtx, "single_view": mtx + 2 * g}
    operators = {}
    for method, matrix in matrices.items():
        beta, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=tolerance)
        operators[method] = {
            "beta": beta.tolist(),
            "rank": int(rank),
            "singular_values": singular.tolist(),
            "eigenvalues": np.linalg.eigvalsh(matrix).tolist(),
            "equation_error": float(np.max(np.abs(matrix @ beta - rhs))),
        }
    return {
        "M": mtx.tolist(),
        "G": g.tolist(),
        "m": rhs.tolist(),
        "operators": operators,
    }


def cold_readings(mu, gamma, views, folds, programs):
    """Produce other-fold readings and conditional partition variance, label-free."""
    mu, gamma, views, folds = map(np.asarray, (mu, gamma, views, folds))
    n, d = mu.shape
    if (
        views.ndim != 4
        or views.shape[0] != n
        or views.shape[2:] != (3, d)
        or gamma.shape != (n, d, d)
        or folds.shape != (n,)
        or set(np.unique(folds)) != {0, 1}
        or not all(np.isfinite(a).all() for a in (mu, gamma, views))
    ):
        raise ValueError(
            "Complete label-free mean/covariance/view/fold contract required"
        )
    targets = len(programs[0]["operators"][METHODS[0]]["beta"][0])
    mean = np.full((3, n, targets), np.nan)
    score = np.full((3, n, views.shape[1], 3, targets), np.nan)
    variance = np.full_like(mean, np.nan)
    if sorted(p["source_fold"] for p in programs) != [0, 1]:
        raise ValueError("Exactly one program per calibration fold required")
    for p in programs:
        held = np.flatnonzero(folds != p["source_fold"])
        for i, method in enumerate(METHODS):
            beta = np.asarray(p["operators"][method]["beta"])
            mean[i, held] = mu[held] @ beta
            score[i, held] = views[held] @ beta
            variance[i, held] = 2 * np.einsum("dq,ude,eq->uq", beta, gamma[held], beta)
    if not all(np.isfinite(a).all() for a in (mean, score, variance)):
        raise ValueError("Incomplete cold routing")
    return {"mean_scores": mean, "view_scores": score, "partition_variance": variance}


def generating_world():
    """Exhaust six iid errors to distinguish latent propensity from finite-bank mean."""
    rows = np.array(list(product([-1.0, 1.0], repeat=7)))
    theta, errors = rows[:, :1], rows[:, 1:]
    q = np.stack([np.ones_like(errors), theta + errors], axis=-1)
    mu, gamma = orbit_moments(q)
    p = construct(mu, gamma, theta)
    return {
        "exhaustive_rows": len(rows),
        "propensity_slope": p["operators"]["orbit_proxy"]["beta"][1][0],
        "finite_bank_slope": p["operators"]["finite_bank"]["beta"][1][0],
        "single_view_slope": p["operators"]["single_view"]["beta"][1][0],
        "M": p["M"],
        "G": p["G"],
        "qualification": "stipulated iid generating errors; not inferred natural personality law",
    }
