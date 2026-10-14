#!/usr/bin/env python3
"""Finite Euclidean dependence and signed source action accounts; no predictors."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_response_function import aligned_source


def finite_rows(values):
    """Require at least two finite vector observations at a declared row grain."""
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("At least two finite nonempty vector rows required")
    return x


def centered_distances(values):
    """Double-center true Euclidean distances, including their zero diagonal."""
    x = finite_rows(values)
    difference = x[:, None, :] - x[None, :, :]
    distance = np.sqrt(np.einsum("ijf,ijf->ij", difference, difference))
    return distance - distance.mean(0) - distance.mean(1)[:, None] + distance.mean()


def distance_account(left, right):
    """Return biased V_n^2 and R_n^2; these are not Pearson correlation."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if (
        a.ndim != 2
        or a.shape != b.shape
        or a.shape[0] != a.shape[1]
        or len(a) < 2
        or not np.isfinite([a, b]).all()
    ):
        raise ValueError("Aligned finite square distance accounts required")
    covariance = float(np.mean(a * b))
    variance_x, variance_y = float(np.mean(a * a)), float(np.mean(b * b))
    denominator = np.sqrt(variance_x * variance_y)
    return {
        "distance_covariance_squared": covariance,
        "distance_variance_x_squared": variance_x,
        "distance_variance_y_squared": variance_y,
        "distance_correlation_squared": float(covariance / denominator)
        if denominator > 0
        else None,
    }


def dependence_accounts(features, responses, groups):
    """Keep pooled and finite production-stratified Euclidean V statistics."""
    x, y = finite_rows(features), finite_rows(responses)
    g = np.asarray(groups)
    if len(x) != len(y) or g.shape != (len(x),):
        raise ValueError("Aligned whole profiles and production blocks required")
    a, b = centered_distances(x), centered_distances(y)
    pieces = []
    for key in np.unique(g):
        ids = np.flatnonzero(g == key)
        if len(ids) < 2:
            raise ValueError("Stratified V statistic requires at least two rows/block")
        pieces.append((ids, centered_distances(x[ids]), centered_distances(y[ids])))
    return a, b, pieces


def stratified_account(pieces, count):
    """Weight all block covariance/variance accounts by their original row mass."""
    if not pieces or count != sum(len(ids) for ids, _, _ in pieces):
        raise ValueError("Complete nonoverlapping production blocks required")
    result = {
        name: 0.0
        for name in (
            "distance_covariance_squared",
            "distance_variance_x_squared",
            "distance_variance_y_squared",
        )
    }
    for ids, a, b in pieces:
        block = distance_account(a, b)
        for key in result:
            result[key] += len(ids) / count * block[key]
    denominator = np.sqrt(
        result["distance_variance_x_squared"] * result["distance_variance_y_squared"]
    )
    result["distance_correlation_squared"] = (
        float(result["distance_covariance_squared"] / denominator)
        if denominator > 0
        else None
    )
    return result


def profile_permutations(groups, count, seed):
    """Shuffle whole row identities inside production blocks; never profile cells."""
    g = np.asarray(groups)
    if g.ndim != 1 or len(g) < 2 or not isinstance(count, int) or count < 1:
        raise ValueError("Row blocks and a positive fixed permutation count required")
    rng = np.random.default_rng(seed)
    result = np.tile(np.arange(len(g)), (count, 1))
    for row in result:
        for key in np.unique(g):
            ids = np.flatnonzero(g == key)
            row[ids] = rng.permutation(ids)
    return result


def permuted_dependence(accounts, permutations):
    """Use one fixed pairing-null family for pooled and blocked distance accounts."""
    a, b, pieces = accounts
    permutations = np.asarray(permutations)
    if permutations.ndim != 2 or permutations.shape[1] != len(a):
        raise ValueError("Complete whole-profile permutation indices required")
    null = np.empty((len(permutations), 2))
    for k, permutation in enumerate(permutations):
        if not np.array_equal(np.sort(permutation), np.arange(len(a))):
            raise ValueError("Permutation must use every row exactly once")
        null[k, 0] = np.mean(a * b[np.ix_(permutation, permutation)])
        blocked = 0.0
        for ids, block_a, block_b in pieces:
            mapped = np.searchsorted(ids, permutation[ids])
            if np.any(mapped >= len(ids)) or not np.array_equal(
                ids[mapped], permutation[ids]
            ):
                raise ValueError("Whole profiles cannot cross production blocks")
            blocked += (
                len(ids) / len(a) * np.mean(block_a * block_b[np.ix_(mapped, mapped)])
            )
        null[k, 1] = blocked
    return null


def source_group_ledger(features, responses, groups, displacements):
    """Expose full/within/between source contributions at one fixed response origin."""
    x, y, g = aligned_source(features, responses, groups)
    d = finite_rows(displacements)
    if d.shape[1] != x.shape[1]:
        raise ValueError("Source and displacement vector dimensions differ")
    keys = np.unique(g)
    origin, base = x.mean(0), y.mean(0)
    weights = d @ (x - origin).T / len(x)
    ledger = np.empty((len(d), 3, len(keys), *y.shape[1:]))
    for j, key in enumerate(keys):
        ids = g == key
        xg, yg = x[ids].mean(0), y[ids].mean(0)
        ledger[:, 0, j] = np.einsum("pe,e...->p...", weights[:, ids], y[ids] - base)
        local_weights = d @ (x[ids] - xg).T / len(x)
        ledger[:, 1, j] = np.einsum("pe,e...->p...", local_weights, y[ids] - yg)
        ledger[:, 2, j] = np.einsum(
            "p,...->p...", len(x[ids]) / len(x) * (d @ (xg - origin)), yg - base
        )
    return keys, weights, ledger
