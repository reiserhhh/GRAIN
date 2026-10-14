"""Affine, conditional, and nonlinear transport contracts for F12."""

from __future__ import annotations

import numpy as np


def affine(value: np.ndarray | float, d: float, c: float) -> np.ndarray:
    """Apply y'=d+c*y."""
    return d + c * np.asarray(value, dtype=float)


def conditional(value: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
    """Apply a condition-dependent affine transport."""
    return np.asarray(z, dtype=float) + (1.0 + .25 * np.asarray(z, dtype=float)) * np.asarray(value, dtype=float)


def cubic(value: np.ndarray | float) -> np.ndarray:
    """Apply a monotone-local cubic perturbation."""
    value = np.asarray(value, dtype=float)
    return value + .1 * value**3


def inverse_affine(value: np.ndarray | float, d: float, c: float) -> np.ndarray:
    """Invert affine transport when c is nonzero."""
    if abs(c) < 1e-12:
        raise ValueError("non-invertible affine slope")
    return (np.asarray(value, dtype=float) - d) / c
