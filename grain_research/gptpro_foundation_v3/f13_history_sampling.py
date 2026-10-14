#!/usr/bin/env python3
"""Joint history draws under a declared fixed-source binary observation law."""

import numpy as np


def sample_histories(
    poles, qualifications, transition, initial, draws, rng, complete=False
):
    """Forward-filter/backward-sample paths; future evidence smooths earlier states.

    These are working latent-history draws, not posterior samples of B2/M model
    coefficients. Refit-on-path estimators differ from online filtered counts.
    """
    p, q = np.asarray(poles, float), np.asarray(qualifications, float)
    transition, initial = (
        np.asarray(transition, float).reshape(2, 2),
        np.asarray(initial, float),
    )
    if (
        len(p) < 2
        or p.shape != q.shape
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
        or np.any((p < 0) | (p > 1) | (q < 0) | (q > 1))
    ):
        raise ValueError("aligned finite observation probabilities required")
    if (
        not np.isfinite(transition).all()
        or np.any((transition < 0) | (transition > 1))
        or initial.shape != (2, 2)
        or not np.isfinite(initial).all()
        or np.any(initial < 0)
        or not np.isclose(initial.sum(), 1)
    ):
        raise ValueError("valid transition and initial law required")
    if complete:
        if not np.all(q == 1):
            raise ValueError("complete panel requires eligibility")
        return (rng.random((draws, len(p))) < p).astype(int)
    likelihood = q[:, None] * np.column_stack([1 - p, p]) + (1 - q[:, None]) / 2
    alpha = initial * likelihood[0, :, None] * likelihood[1, None, :]
    if alpha.sum() <= 0:
        raise ValueError("impossible initial observations")
    alpha /= alpha.sum()
    forward = [None, alpha.copy()]
    probabilities = np.stack([1 - transition, transition], axis=-1)
    for t in range(2, len(p)):
        alpha = (alpha[:, :, None] * probabilities * likelihood[t]).sum(axis=0)
        if alpha.sum() <= 0:
            raise ValueError("impossible observation")
        alpha /= alpha.sum()
        forward.append(alpha.copy())
    paths = np.empty((draws, len(p)), dtype=int)
    last = rng.choice(4, size=draws, p=alpha.ravel())
    paths[:, -2], paths[:, -1] = last // 2, last % 2
    for t in range(len(p) - 3, -1, -1):
        b, y = paths[:, t + 1], paths[:, t + 2]
        weights = np.stack(
            [forward[t + 1][a, b] * probabilities[a, b, y] for a in (0, 1)], axis=1
        )
        prob = weights[:, 1] / weights.sum(axis=1)
        paths[:, t] = rng.random(draws) < prob
    return paths


def path_counts(paths, qualifications):
    """Weighted transition sufficient statistics for each complete drawn path."""
    paths = np.asarray(paths)
    q = np.asarray(qualifications, float)
    if paths.ndim != 2 or paths.shape[1] != len(q) or not np.isin(paths, [0, 1]).all():
        raise ValueError("aligned binary histories required")
    result = np.zeros((len(paths), 4, 2))
    for t in range(2, paths.shape[1]):
        np.add.at(
            result,
            (np.arange(len(paths)), 2 * paths[:, t - 2] + paths[:, t - 1], paths[:, t]),
            q[t],
        )
    return result
