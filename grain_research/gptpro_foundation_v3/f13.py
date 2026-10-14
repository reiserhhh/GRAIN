"""Uncertainty decomposition and minimum detectable change for F13."""

from __future__ import annotations

import numpy as np


def total_variance(components: dict[str, float]) -> float:
    """Return the sum of declared independent variance components."""
    return float(sum(float(value) for value in components.values()))


def paired_change_variance(var_a: float, var_b: float, covariance: float) -> float:
    """Return Var(A-B)."""
    return float(max(0.0, var_a + var_b - 2.0 * covariance))


def mdc(variance: float, z: float = 1.96) -> float:
    """Return a descriptive minimum detectable change."""
    return float(z * np.sqrt(max(0.0, variance)))


def gaussian_interval(value: float, variance: float, z: float = 1.96) -> tuple[float, float]:
    """Return a Gaussian plug-in interval."""
    half = z * np.sqrt(max(0.0, variance))
    return float(value - half), float(value + half)
