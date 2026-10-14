#!/usr/bin/env python3
"""Declared Laplace parameter mixture for F05, distinct from exact posterior claims."""

import numpy as np

from grain_research.gptpro_foundation_v3.f05_profiles import central_hessian
from grain_research.gptpro_foundation_v3.f05_recovery import (
    arrays,
    filter_arrays,
    fit_person,
)


def laplace_person(rows, source_model):
    """Use the frozen MAP fitter and a 1e-4 central Hessian, without PSD rescue.

    A Gaussian approximation in (a,b,eta) is a declared new numerical layer, not
    an exact Bayesian posterior. Nonconvergence/nonpositive curvature is returned
    explicitly and may not be hidden by flooring eigenvalues.
    """
    prior = source_model["prior"]
    logs = np.array(source_model["log_sigma"])
    fit = fit_person(rows, logs, prior)
    if not fit["success"]:
        return {"status": "MAP_FAILED", "fit": fit}
    observed = arrays(rows)
    precision = np.linalg.inv(prior["Sigma"])
    mu = np.asarray(prior["mu"])

    def objective(psi):
        delta = psi - mu
        return (
            -filter_arrays(psi, *observed, logs)["loglik"]
            + 0.5 * delta @ precision @ delta
        )

    hessian = central_hessian(objective, fit["psi"], step=1e-4)
    eigen = np.linalg.eigvalsh(hessian)
    if not np.isfinite(hessian).all() or eigen.min() <= 0:
        return {
            "status": "CURVATURE_UNSUPPORTED",
            "fit": fit,
            "eigenvalues": eigen.tolist(),
        }
    return {
        "status": "LAPLACE_APPROXIMATION",
        "fit": fit,
        "hessian": hessian.tolist(),
        "parameter_covariance": np.linalg.inv(hessian).tolist(),
        "eigenvalues": eigen.tolist(),
    }


def filter_parameter_batch(parameters, rows, log_sigma):
    """Vectorize independent exact scalar filters over parameter draws."""
    p = np.asarray(parameters, float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("finite parameter draws required")
    y, z, times = arrays(rows)
    rho = 0.99 * np.tanh(p[:, 2])
    q, r = np.exp(2 * np.asarray(log_sigma))
    stationary = q / (1 - rho * rho)
    mean = np.zeros(len(p))
    variance = stationary.copy()
    for t in range(len(y)):
        if t:
            coefficient = rho ** int(times[t] - times[t - 1])
            mean *= coefficient
            variance = coefficient**2 * variance + stationary * (1 - coefficient**2)
        gain = variance / (variance + r)
        mean += gain * (y[t] - p[:, 0] - p[:, 1] * z[t] - mean)
        variance *= 1 - gain
    return mean, variance


def conditional_target_components(parameters, rows, log_sigma, future_conditions):
    """Return three target means/covariances for each parameter draw."""
    p = np.asarray(parameters, float)
    z = np.asarray(future_conditions, float)
    if z.shape != (4,) or not np.isfinite(z).all():
        raise ValueError("four planned future conditions required")
    state, variance = filter_parameter_batch(p, rows, log_sigma)
    rho = 0.99 * np.tanh(p[:, 2])
    q, r = np.exp(2 * np.asarray(log_sigma))
    total = np.zeros((len(p), 3, 3))
    average_loading = sum(rho**k for k in range(1, 5)) / 4
    state_loading = np.column_stack([np.ones(len(p)), rho**4, average_loading])
    total += (
        variance[:, None, None] * state_loading[:, :, None] * state_loading[:, None, :]
    )
    for j in range(1, 5):
        loading = np.column_stack(
            [np.zeros(len(p)), rho ** (4 - j), sum(rho**l for l in range(5 - j)) / 4]
        )
        total += q * loading[:, :, None] * loading[:, None, :]
        error_loading = np.array([0.0, float(j == 4), 0.25])
        total += r * np.outer(error_loading, error_loading)
    means = np.column_stack(
        [
            state,
            p[:, 0] + p[:, 1] * z[-1] + rho**4 * state,
            p[:, 0] + p[:, 1] * z.mean() + average_loading * state,
        ]
    )
    return means, total


def parameter_mixture(parameters, rows, log_sigma, future_conditions):
    """Law of total covariance across the supplied finite parameter sample."""
    means, covariances = conditional_target_components(
        parameters, rows, log_sigma, future_conditions
    )
    center = means.mean(axis=0)
    between = (means - center).T @ (means - center) / len(means)
    within = covariances.mean(axis=0)
    return {
        "mean": center,
        "event_state": within,
        "parameter": between,
        "total": within + between,
    }
