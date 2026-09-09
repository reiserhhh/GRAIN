#!/usr/bin/env python3
"""Order-sensitive GRAIN descriptors and author-held channel accounting."""

from __future__ import annotations

from itertools import permutations

import numpy as np


def path_descriptors(path, *, groups=((0, 1, 2, 3), (4, 5, 6, 7))):
    """Separate dispersion from order; never infer equal-time process parameters."""
    x = np.asarray(path, float)
    if x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("At least four finite vector observations required")
    if sorted(i for g in groups for i in g) != list(range(x.shape[1])):
        raise ValueError("Groups must partition the coordinate columns")
    variance = x.var(0)
    sd = np.sqrt(variance)
    expected = 2 * len(x) / (len(x) - 1) * variance
    roughness = np.mean(np.diff(x, axis=0) ** 2, axis=0)
    excess = (
        np.divide(roughness, expected, out=np.ones_like(sd), where=expected > 1e-20) - 1
    )
    z = (x - x.mean(0)) / np.where(sd > 1e-10, sd, 1)
    area = [
        np.mean(z[:-1, j] * z[1:, k] - z[:-1, k] * z[1:, j])
        for g in groups
        for n, j in enumerate(g)
        for k in g[n + 1 :]
    ]
    order = np.r_[(x[-1] - x[0]) / (len(x) - 1), excess, area]
    return sd, order


def residualize_descriptive(values, context):
    """OLS adjustment for descriptive associations, never a prediction transform."""
    y, z = np.asarray(values, float), np.asarray(context, float)
    if (
        y.shape[0] != len(z)
        or z.ndim != 2
        or not np.isfinite(y).all()
        or not np.isfinite(z).all()
    ):
        raise ValueError("Aligned finite values and context required")
    z = np.c_[np.ones(len(z)), z]
    return y - z @ np.linalg.lstsq(z, y, rcond=None)[0]


def column_correlations(x, y):
    """Pearson correlation for each column, with explicit constant-coordinate NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("Aligned matrix and vector required")
    xc, yc = x - x.mean(0), y - y.mean()
    denom = np.sqrt(np.sum(xc**2, 0) * np.sum(yc**2))
    return np.divide(
        xc.T @ yc, denom, out=np.full(x.shape[1], np.nan), where=denom > 1e-14
    )


def clustered_links(x, y, *, draws=499, permutation_draws=None, seed=20260910):
    """Resample/permutate whole authors, preserving repeated transition pairing."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 3 or y.shape != x.shape[:2] or len(x) < 8:
        raise ValueError("Author by transition by coordinate arrays required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite matched author arrays required")
    # Remove common transition offsets, without eliminating author-specific motion.
    x, y = x - x.mean(0), y - y.mean(0)
    observed = column_correlations(x.reshape(-1, x.shape[2]), y.ravel())
    permutation_draws = draws if permutation_draws is None else permutation_draws
    rng, boot, null = np.random.default_rng(seed), [], []
    for _ in range(draws):
        ix = rng.integers(0, len(x), len(x))
        boot.append(column_correlations(x[ix].reshape(-1, x.shape[2]), y[ix].ravel()))
    for _ in range(permutation_draws):
        ix = rng.permutation(len(x))
        null.append(column_correlations(x.reshape(-1, x.shape[2]), y[ix].ravel()))
    boot, null = np.asarray(boot), np.asarray(null)
    lo, hi = np.nanquantile(boot, (0.025, 0.975), axis=0)
    p = (1 + np.sum(np.abs(null) >= np.abs(observed), axis=0)) / (permutation_draws + 1)
    p[~np.isfinite(observed)] = np.nan
    return observed, lo, hi, p


def shapley_mse(mse, *, blocks=3):
    """Allocate held-out gain exactly over all block orders; allow negative values."""
    if set(mse) != set(range(2**blocks)) or not np.isfinite(list(mse.values())).all():
        raise ValueError("All finite subset losses required")
    values, count = np.zeros(blocks), 0
    for order in permutations(range(blocks)):
        mask = 0
        for j in order:
            next_mask = mask | (1 << j)
            values[j] += mse[mask] - mse[next_mask]
            mask = next_mask
        count += 1
    values /= count
    if not np.isclose(values.sum(), mse[0] - mse[2**blocks - 1], atol=1e-12):
        raise AssertionError("Gain accounting does not close")
    return values
