#!/usr/bin/env python3
"""Compile a source SwiGLU/RMS operator and readout into one fixed covector."""

import numpy as np

from .gated_update import coefficients


def source_readout(x, t, weights):
    """Return N(x)^T B(x)^T t without constructing a dense hidden-by-hidden Jacobian."""
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    p, q = coefficients(x, weights)
    output_weights = weights["down"].T @ t
    m = weights["up"].T @ (p * output_weights) + weights["gate"].T @ (
        q * output_weights
    )
    gamma_m = weights["gamma"] * m
    rho = np.sqrt(np.mean(x * x) + float(weights["epsilon"]))
    return gamma_m / rho - x * np.dot(x, gamma_m) / (x.size * rho**3)


def mean_readout(source_x, t, weights):
    """Average paired source operators; no current x, increment, or target is needed."""
    if len(source_x) == 0:
        raise ValueError("At least one independent source context is required")
    return np.mean([source_readout(x, t, weights) for x in source_x], axis=0)
