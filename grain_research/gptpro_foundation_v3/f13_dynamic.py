#!/usr/bin/env python3
"""Joint current-state and future-observation uncertainty, conditional on parameters."""

import numpy as np

from grain_research.gptpro_foundation_v3.f13_uncertainty import validate_covariance


def joint_forecast(psi, log_sigma, state_mean, state_variance, future_conditions):
    """Return law of (s_T, y_T+1,...,y_T+h) using shared innovation coefficients.

    Parameters are fixed in this function. Source/individual parameter uncertainty
    must be added by an outer mixture, never by labelling this covariance complete.
    """
    psi = np.asarray(psi, float)
    logs = np.asarray(log_sigma, float)
    z = np.asarray(future_conditions, float)
    if (
        psi.shape != (3,)
        or logs.shape != (2,)
        or z.ndim != 1
        or not len(z)
        or not np.isfinite(np.r_[psi, logs, z, state_mean, state_variance]).all()
        or state_variance < 0
    ):
        raise ValueError("finite parameters and nonnegative state variance required")
    a, b, eta = psi
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * logs)
    h = len(z)
    # Columns: terminal state, h future innovations, h observation errors.
    loading = np.zeros((h + 1, 1 + 2 * h))
    loading[0, 0] = 1
    mean = np.r_[state_mean, a + b * z + rho ** np.arange(1, h + 1) * state_mean]
    for k in range(1, h + 1):
        loading[k, 0] = rho**k
        for j in range(1, k + 1):
            loading[k, j] = rho ** (k - j)
        loading[k, h + k] = 1
    variances = np.r_[state_variance, np.full(h, q), np.full(h, r)]
    covariance = validate_covariance((loading * variances) @ loading.T)
    return mean, covariance, loading, variances


def target_projection(horizon=4):
    """Current latent state, final noisy observation, mean of future observations."""
    if horizon < 1:
        raise ValueError("positive horizon required")
    matrix = np.zeros((3, horizon + 1))
    matrix[0, 0] = matrix[1, -1] = 1
    matrix[2, 1:] = 1 / horizon
    return matrix
