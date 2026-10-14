#!/usr/bin/env python3
"""Positive-law witnesses for a declared summary, not predictor optimization."""

from __future__ import annotations

import numpy as np


def conditional_contrasts(strata):
    """Disjoint four-corner columns preserve each stratum's b/c margins."""
    if not isinstance(strata, int) or strata < 1:
        raise ValueError("Positive integer stratum count required")
    return np.kron(np.eye(strata), np.array([1.0, -1.0, -1.0, 1.0])[:, None])


def summary_columns(coordinates, second_moments=False):
    """Retain all unique raw second moments if declared; never select a subset."""
    g = np.asarray(coordinates, float)
    if g.ndim != 2 or not np.isfinite(g).all():
        raise ValueError("Finite event-by-coordinate matrix required")
    if not second_moments:
        return g
    i, j = np.triu_indices(g.shape[1])
    return np.column_stack([g, g[:, i] * g[:, j]])


def conditional_null_space(coordinates, contrasts, rtol=1e-10):
    """Find the complete summary-annihilating conditional-law space once."""
    g, d = np.asarray(coordinates, float), np.asarray(contrasts, float)
    if g.ndim != 2 or d.ndim != 2 or len(g) != len(d):
        raise ValueError("Matched event rows required")
    if not np.isfinite(g).all() or not np.isfinite(d).all() or rtol <= 0:
        raise ValueError("Finite matrices and positive fixed cutoff required")
    matrix = (d.T @ g).T
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    cutoff = rtol * singular[0] if len(singular) else 0
    rank = int(np.count_nonzero(singular > cutoff))
    return {
        "rank": rank,
        "nullity": d.shape[1] - rank,
        "singular_values": singular,
        "cutoff": cutoff,
        "basis": vh[rank:].T,
        "matrix": matrix,
    }


def positive_witness(values, contrasts, basis, tolerance=1e-10):
    """Construct one strictly positive pair per target; never reinterpret it as people."""
    y, d, n = (
        np.asarray(values, float),
        np.asarray(contrasts, float),
        np.asarray(basis, float),
    )
    if y.ndim != 1 or len(y) != len(d) or n.shape[0] != d.shape[1]:
        raise ValueError("Matched target/contrast/null-space dimensions required")
    vector = d.T @ y
    residual = n @ (n.T @ vector)
    norm = float(np.linalg.norm(residual))
    if norm <= tolerance:
        return {
            "status": "NO_NONZERO_WITNESS_IN_DECLARED_SUBSPACE",
            "residual_norm": norm,
            "plus": None,
            "minus": None,
            "difference": None,
            "alpha": None,
            "epsilon": None,
        }
    alpha = residual / norm
    direction = d @ alpha
    epsilon = 0.5 / (len(y) * np.max(abs(direction)))
    plus, minus = 1 / len(y) + epsilon * direction, 1 / len(y) - epsilon * direction
    return {
        "status": "POSITIVE_CONDITIONAL_MEAN_FIBER",
        "residual_norm": norm,
        "plus": plus,
        "minus": minus,
        "difference": float((plus - minus) @ y),
        "alpha": alpha,
        "epsilon": float(epsilon),
    }
