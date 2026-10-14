"""Fixed-noise F05 likelihood profiles, with analytic nuisance elimination."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from grain_research.gptpro_foundation_v3.f05 import eta_from_rho
from grain_research.gptpro_foundation_v3.f05_recovery import arrays


def profile_point(rows, rho, log_sigma, prior=None):
    """Minimize over a,b at fixed rho using GLS or conditional Gaussian MAP.

    Shared noise and prior are fixed source estimates. Values are conditional
    likelihood diagnostics, not noise-profiled likelihood or calibrated CIs.
    """
    if not 0 <= rho < 0.99:
        raise ValueError("rho outside frozen positive profile domain")
    y, z, times = arrays(rows)
    x = np.column_stack((np.ones(len(y)), z))
    q, r = np.exp(2 * np.asarray(log_sigma))
    covariance = q / (1 - rho * rho) * rho ** np.abs(
        times[:, None] - times[None, :]
    ) + r * np.eye(len(y))
    chol = cho_factor(covariance, lower=True)
    vx, vy = cho_solve(chol, x), cho_solve(chol, y)
    lhs, rhs = x.T @ vx, x.T @ vy
    eta = eta_from_rho(rho)
    if prior is not None:
        precision = np.linalg.inv(prior["Sigma"])
        mu = np.asarray(prior["mu"])
        lhs = lhs + precision[:2, :2]
        rhs = rhs + precision[:2, :2] @ mu[:2] - precision[:2, 2] * (eta - mu[2])
    beta = np.linalg.solve(lhs, rhs)
    residual = y - x @ beta
    nll = 0.5 * (
        len(y) * np.log(2 * np.pi)
        + 2 * np.log(np.diag(chol[0])).sum()
        + residual @ cho_solve(chol, residual)
    )
    penalty = 0.0
    if prior is not None:
        delta = np.r_[beta, eta] - mu
        penalty = 0.5 * delta @ precision @ delta
    return {
        "rho": float(rho),
        "psi": [float(beta[0]), float(beta[1]), float(eta)],
        "negative_loglik": float(nll),
        "prior_penalty": float(penalty),
        "objective": float(nll + penalty),
    }


def central_hessian(function, point, step=1e-4):
    """Symmetric central difference Hessian in the explicit a,b,eta coordinates."""
    point = np.asarray(point, dtype=float)
    dimension = len(point)
    matrix = np.zeros((dimension, dimension))
    base = function(point)
    for i in range(dimension):
        ei = np.eye(dimension)[i] * step
        matrix[i, i] = (
            function(point + ei) - 2 * base + function(point - ei)
        ) / step**2
        for j in range(i):
            ej = np.eye(dimension)[j] * step
            value = (
                function(point + ei + ej)
                - function(point + ei - ej)
                - function(point - ei + ej)
                + function(point - ei - ej)
            ) / (4 * step**2)
            matrix[i, j] = matrix[j, i] = value
    return matrix
