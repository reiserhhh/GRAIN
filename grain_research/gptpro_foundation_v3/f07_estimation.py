"""Exact categorical F07 MAP fitting; observed counts only enter estimation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .f07_channels import binary_law, categorical_information, ordinal_law

PRIOR_MEAN = np.array([0.0, 0.5])
PRIOR_VARIANCE = np.array([1.0, 0.25])


def objective_gradient(theta, counts, presence, flip, variant="base", ordinal=False):
    """Negative log posterior and exact derivative, retaining missing outcomes."""
    counts = np.asarray(counts, dtype=float)
    categories = 5 if ordinal else 3
    if (
        counts.shape != (2, categories)
        or not np.isfinite(counts).all()
        or (counts < 0).any()
    ):
        raise ValueError("finite nonnegative counts for z=-1,+1 required")
    displacement = np.asarray(theta) - PRIOR_MEAN
    value = 0.5 * np.sum(displacement**2 / PRIOR_VARIANCE)
    gradient = displacement / PRIOR_VARIANCE
    for row, z in zip(counts, (-1.0, 1.0), strict=True):
        probability, derivative = (
            ordinal_law(theta, z)
            if ordinal
            else binary_law(theta, z, presence, flip, variant)
        )
        occupied = row > 0
        if (probability[occupied] <= 0).any():
            return float("inf"), np.full(2, np.nan)
        value -= np.dot(row[occupied], np.log(probability[occupied]))
        gradient = gradient - np.sum(
            row[occupied, None] * derivative[occupied] / probability[occupied, None],
            axis=0,
        )
    return float(value), gradient


def fit_channel(counts, presence, flip, variant="base", ordinal=False):
    """Frozen BFGS MAP with expected-information covariance, not exact posterior."""
    # Validate before optimization, even when the optimizer might stop immediately.
    objective_gradient(PRIOR_MEAN, counts, presence, flip, variant, ordinal)
    result = minimize(
        objective_gradient,
        PRIOR_MEAN.copy(),
        args=(counts, presence, flip, variant, ordinal),
        method="BFGS",
        jac=True,
        options={"gtol": 1e-8, "maxiter": 200},
    )
    information = np.diag(1 / PRIOR_VARIANCE)
    for row, z in zip(np.asarray(counts), (-1.0, 1.0), strict=True):
        law = (
            ordinal_law(result.x, z)
            if ordinal
            else binary_law(result.x, z, presence, flip, variant)
        )
        information += row.sum() * categorical_information(*law)
    return {
        "estimate": result.x,
        "covariance": np.linalg.inv(information),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "gradient_norm": float(np.linalg.norm(result.jac, ord=np.inf)),
        "objective": float(result.fun),
    }
