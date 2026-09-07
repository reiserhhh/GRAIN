#!/usr/bin/env python3
"""Conditional joint-moment approximations to an exact finite attention readout."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from grain_research.vector_mechanisms.query_decomposition import centered_values
from grain_research.vector_mechanisms.query_tilt import validate_arrays


def conditional_moment_readouts(keys, values, query, groups):
    """Separate comment attention mass from within-comment key/value response.

    All moments are empirical within the declared groups. The Gaussian arm is
    a second-cumulant approximation, not an assumption imposed on the data.
    Outside-group tokens retain their exact normalization contribution.
    """
    k, v, q = validate_arrays(keys, values, query)
    centered = centered_values(v, groups)
    if not groups:
        raise ValueError("At least one comment group required")
    logits = k @ q
    normalizer = logsumexp(logits)
    exact = np.exp(logits - normalizer) @ centered
    uniform, weighted = np.zeros(v.shape[1]), np.zeros(v.shape[1])
    gaussian_logs, covariances = [], []
    outside = np.ones(len(k), dtype=bool)
    for g in groups:
        ix = np.asarray(g)
        outside[ix] = False
        l = logits[ix]
        covariance = centered[ix].T @ (l - l.mean()) / len(ix)
        uniform += len(ix) / len(k) * covariance
        weighted += np.exp(logsumexp(l) - normalizer) * covariance
        gaussian_logs.append(np.log(len(ix)) + l.mean() + 0.5 * l.var())
        covariances.append(covariance)
    external_log = logsumexp(logits[outside]) if outside.any() else -np.inf
    gaussian_normalizer = logsumexp([*gaussian_logs, external_log])
    gaussian = np.exp(np.array(gaussian_logs) - gaussian_normalizer) @ np.array(
        covariances
    )
    return {
        "exact": exact,
        "uniform_moment": uniform,
        "exact_mass_moment": weighted,
        "gaussian_mass_moment": gaussian,
    }
