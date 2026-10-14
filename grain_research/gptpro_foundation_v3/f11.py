"""Multi-trait multi-method measurement-model certificates for F11."""

from __future__ import annotations

import numpy as np


TRAITS = ("T1", "T2", "T3")
METHODS = ("grain", "nli", "reader")
OCCASIONS = ("o1", "o2")


def covariance_matrix(trait_var: float = 1.0, method_var: float = .2, occasion_var: float = .1, error_var: float = .3, shared_actor: float = .1) -> np.ndarray:
    """Construct a declared 3x3x2 covariance proxy."""
    size = len(TRAITS) * len(METHODS) * len(OCCASIONS)
    matrix = np.eye(size) * (trait_var + method_var + occasion_var + error_var)
    for i in range(size):
        for j in range(size):
            if i != j and i // (len(METHODS) * len(OCCASIONS)) == j // (len(METHODS) * len(OCCASIONS)):
                matrix[i, j] += shared_actor
    return matrix


def fit_moments(matrix: np.ndarray) -> dict:
    """Return diagonal/shared components with an explicit anchor status."""
    diagonal = np.diag(matrix); off = matrix - np.diag(diagonal)
    return {"diagonal_mean": float(np.mean(diagonal)), "off_diagonal_mean": float(np.mean(off)), "rank": int(np.linalg.matrix_rank(matrix)), "anchor_required": True}


def shared_method_covariance(matrix: np.ndarray, block_size: int = 6) -> float:
    """Estimate a descriptive same-method off-diagonal component."""
    if matrix.size == 0:
        return float("nan")
    return float(np.mean(matrix[:block_size, :block_size] - np.diag(np.diag(matrix[:block_size, :block_size]))))
