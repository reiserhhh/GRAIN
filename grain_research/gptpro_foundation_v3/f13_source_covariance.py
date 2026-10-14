#!/usr/bin/env python3
"""Source covariance-component regression with a declared residual kernel."""

import numpy as np


def covariance_basis(x, residual_kernel):
    """Four linearly independent bases: level, covariance, slope and error."""
    x = np.asarray(x, float)
    c = np.asarray(residual_kernel, float)
    if x.ndim != 2 or x.shape[1] != 2 or c.shape != (len(x), len(x)):
        raise ValueError("aligned two-column design and covariance kernel required")
    bases = np.array(
        [
            np.outer(x[:, 0], x[:, 0]),
            np.outer(x[:, 0], x[:, 1]) + np.outer(x[:, 1], x[:, 0]),
            np.outer(x[:, 1], x[:, 1]),
            c,
        ]
    )
    if np.linalg.matrix_rank(bases.reshape(4, -1).T) != 4:
        raise ValueError("source design cannot identify four covariance components")
    return bases


def fit_source_covariance(values, x, residual_kernel):
    """Fit source moments only; invalid variance estimates are retained, not floored."""
    values = np.asarray(values, float)
    if (
        values.ndim != 2
        or len(values) < 3
        or not np.isfinite(values).all()
        or values.shape[1] != len(x)
    ):
        raise ValueError("three finite aligned source authors required")
    bases = covariance_basis(x, residual_kernel)
    empirical = np.cov(values, rowvar=False)
    coefficients = np.linalg.lstsq(
        bases.reshape(4, -1).T, empirical.ravel(), rcond=None
    )[0]
    prior = np.array(
        [[coefficients[0], coefficients[1]], [coefficients[1], coefficients[2]]]
    )
    eigenvalues = np.linalg.eigvalsh(prior)
    valid = bool(eigenvalues.min() > 0 and coefficients[3] > 0)
    return {
        "mean": np.linalg.lstsq(x, values.mean(axis=0), rcond=None)[0].tolist(),
        "prior": prior.tolist(),
        "sigma2": float(coefficients[3]),
        "coefficients": coefficients.tolist(),
        "prior_eigenvalues": eigenvalues.tolist(),
        "status": "SUPPORTED" if valid else "INVALID_MOMENT_VARIANCE",
        "covariance_residual_frobenius": float(
            np.linalg.norm(empirical - np.einsum("k,kij->ij", coefficients, bases))
        ),
        "source_authors": len(values),
    }
