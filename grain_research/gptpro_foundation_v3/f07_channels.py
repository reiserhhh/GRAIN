"""Frozen F07 channels: opportunity, observation error, cost and missingness."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, ndtr

OPPORTUNITIES = np.array([1, 1, 1, 3, 6])
COSTS = np.array([4, 6, 12, 40, 80])
CUTPOINTS = np.array([-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf])


def binary_law(theta, z, presence, flip, variant="base"):
    """Return [missing,positive,negative] probabilities and a/b derivatives."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,) or not np.isfinite(theta).all() or not np.isfinite(z):
        raise ValueError("finite two-parameter model required")
    if not 0 <= presence <= 1 or not 0 <= flip <= 0.5:
        raise ValueError("invalid presence/flip probability")
    if variant not in ("base", "offtarget", "mnar"):
        raise ValueError("unknown channel variant")
    p = expit(theta[0] + theta[1] * z)
    dp = p * (1 - p) * np.array([1.0, z])
    observed = flip + (1 - 2 * flip) * p
    d_observed = (1 - 2 * flip) * dp
    if variant in ("offtarget", "mnar"):
        observed = 0.8 * observed + 0.1
        d_observed *= 0.8
    c, dc = presence, np.zeros(2)
    if variant == "mnar":
        c = presence * (0.5 + 0.5 * p)
        dc = 0.5 * presence * dp
    probabilities = np.array([1 - c, c * observed, c * (1 - observed)])
    derivative = np.vstack(
        [-dc, dc * observed + c * d_observed, dc * (1 - observed) - c * d_observed]
    )
    return probabilities, derivative


def ordinal_law(theta, z):
    """Separate complete-observation five-category Gaussian reference, SD=1."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,) or not np.isfinite(theta).all() or not np.isfinite(z):
        raise ValueError("finite two-parameter model required")
    shifted = CUTPOINTS - theta[0] - theta[1] * z
    probabilities = np.diff(ndtr(shifted))
    density = np.exp(-0.5 * shifted**2) / np.sqrt(2 * np.pi)
    derivative_mu = density[:-1] - density[1:]
    derivative = derivative_mu[:, None] * np.array([1.0, z])[None, :]
    return probabilities, derivative


def categorical_information(probabilities, derivative):
    """Full categorical Fisher matrix without treating missingness as zero score."""
    probabilities = np.asarray(probabilities)
    derivative = np.asarray(derivative)
    active = probabilities > 0
    if np.any(np.abs(derivative[~active]) > 1e-12):
        raise ValueError("nonregular zero-probability category")
    return derivative[active].T @ (derivative[active] / probabilities[active, None])


def efficient_slope_information(matrix):
    """Eliminate the jointly unknown intercept, rather than quoting I_bb alone."""
    matrix = np.asarray(matrix)
    if matrix[0, 0] <= 0:
        return None
    return float(matrix[1, 1] - matrix[0, 1] ** 2 / matrix[0, 0])


def world_parameters(world):
    """Return five declared channels; flat is calibrated at (a,b)=(0,.5)."""
    if world == "U":
        return np.array([0.01, 0.03, 0.1, 0.5, 1.0]), np.full(5, 0.02)
    if world == "D":
        return np.full(5, 0.9), np.array([0.02, 0.05, 0.12, 0.25, 0.4])
    if world == "P":
        return np.array([0.05, 0.2, 0.6, 0.9, 0.9]), np.array(
            [0.02, 0.04, 0.1, 0.25, 0.45]
        )
    if world != "flat":
        raise ValueError("unknown world")
    errors = np.array([0.02, 0.05, 0.1, 0.15, 0.2])
    per_token = []
    for level, error in enumerate(errors):
        p, d = binary_law([0.0, 0.5], 1.0, 1.0, error)
        per_token.append(
            categorical_information(p, d)[1, 1] * OPPORTUNITIES[level] / COSTS[level]
        )
    values = np.asarray(per_token)
    return 0.5 * values.min() / values, errors


def channel_information(theta, level, world, variant="base"):
    """Balanced +/-1 condition information, per opportunity and per token."""
    if level not in range(5):
        raise ValueError("level must be zero-based 0..4")
    presence, flip = world_parameters(world)
    matrix = (
        sum(
            categorical_information(
                *binary_law(theta, z, presence[level], flip[level], variant)
            )
            for z in (-1.0, 1.0)
        )
        / 2
    )
    return {
        "per_opportunity": matrix,
        "per_token": matrix * OPPORTUNITIES[level] / COSTS[level],
        "presence": float(presence[level]),
        "flip": float(flip[level]),
    }
