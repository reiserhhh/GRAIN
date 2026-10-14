"""Finite action signatures and local quadratic metric for F10."""

from __future__ import annotations

import numpy as np


def edit_signature(base: np.ndarray, edited: np.ndarray) -> np.ndarray:
    """Return a finite edit signature; no continuous Jacobian is implied."""
    return np.asarray(edited, dtype=float) - np.asarray(base, dtype=float)


def quadratic_response(h: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    """Return q(h)=u'h + .5 h'Bh."""
    return float(u @ h + 0.5 * h @ b @ h)


def common_metric(jacobians: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    """Return G=sum_q w J_q'J_q."""
    if not jacobians:
        return np.zeros((0, 0))
    weights = np.ones(len(jacobians)) / len(jacobians) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    return sum(float(w) * j.T @ j for w, j in zip(weights, jacobians, strict=True))


def svd_rank(matrix: np.ndarray, tol: float | None = None) -> int:
    """Return numerical rank with an explicit SVD tolerance."""
    values = np.linalg.svd(matrix, compute_uv=False)
    if not len(values):
        return 0
    threshold = (max(matrix.shape) * np.finfo(float).eps * values[0]) if tol is None else tol
    return int(np.sum(values > threshold))
