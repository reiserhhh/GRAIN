#!/usr/bin/env python3
"""Numerical helpers for a bounded local action-metric diagnostic.

The diagnostic treats a reader's local scalar score as a row Jacobian ``J`` and
fits a source-only operator for text-coordinate changes.  It deliberately uses
the dual (sample-space) ridge solve, so it never materializes a dense
``(hidden_dim * input_dim)^2`` matrix.
"""

from __future__ import annotations

import numpy as np


def standardize_inputs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return source-scaled inputs and the positive per-coordinate scale."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite two-dimensional array")
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return x / scale, scale


def fit_euclidean_action(x: np.ndarray, target: np.ndarray, alpha: float = 1.0) -> dict:
    """Fit ordinary ridge from scaled coordinate changes to vector actions."""
    x, scale = standardize_inputs(x)
    target = np.asarray(target, dtype=float)
    if target.ndim != 2 or len(x) != len(target) or not np.isfinite(target).all():
        raise ValueError("target must align with finite x")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    weight = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ target)
    return {"scale": scale, "weight": weight, "alpha": float(alpha)}


def predict_euclidean(model: dict, x: np.ndarray) -> np.ndarray:
    """Predict vector actions with an ordinary fitted action model."""
    return np.asarray(x, dtype=float) / model["scale"] @ model["weight"]


def fit_action_weighted(
    x: np.ndarray,
    target: np.ndarray,
    jacobian: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """Fit ``W`` under ``sum_i (J_i(target_i-Wx_i))^2 + alpha||W||^2``.

    ``target`` and each Jacobian row are one vector per source observation.  The
    Kronecker design is only ``n x (input_dim * action_dim)``; the solve uses
    its ``n x n`` Gram matrix and therefore remains matrix-free with respect to
    the large action dimension.
    """
    x, scale = standardize_inputs(x)
    target = np.asarray(target, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    if (
        target.ndim != 2
        or jacobian.ndim != 2
        or target.shape != jacobian.shape
        or len(x) != len(target)
        or not np.isfinite(target).all()
        or not np.isfinite(jacobian).all()
    ):
        raise ValueError("x, target and jacobian must be aligned finite matrices")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    design = np.asarray([np.kron(row_x, row_j) for row_x, row_j in zip(x, jacobian)])
    gram = design @ design.T
    dual = np.linalg.solve(
        gram + alpha * np.eye(len(design)),
        np.array([j @ y for j, y in zip(jacobian, target)]),
    )
    vector = design.T @ dual
    action_dim = target.shape[1]
    input_dim = x.shape[1]
    weight = vector.reshape(input_dim, action_dim).T
    residual = design @ vector - np.array([j @ y for j, y in zip(jacobian, target)])
    return {
        "scale": scale,
        "weight": weight,
        "alpha": float(alpha),
        "source_loss": float(np.sum(residual**2) + alpha * np.sum(weight**2)),
        "source_residual_rmse": float(np.sqrt(np.mean(residual**2))),
    }


def predict_weighted(model: dict, x: np.ndarray) -> np.ndarray:
    """Predict a vector action from a weighted operator."""
    return (
        np.asarray(model["weight"]) @ (np.asarray(x, dtype=float) / model["scale"]).T
    ).T


def scalar_action(jacobian: np.ndarray, action: np.ndarray) -> float:
    """Evaluate the first-order scalar action ``J action``."""
    return float(np.asarray(jacobian, dtype=float) @ np.asarray(action, dtype=float))


def paired_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    """Return RMSE, Pearson r and sign agreement for finite paired effects."""
    p, y = np.asarray(predicted, dtype=float), np.asarray(target, dtype=float)
    if (
        p.shape != y.shape
        or p.ndim != 1
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("paired one-dimensional finite arrays required")
    if len(p) > 2 and np.std(p) > 1e-12 and np.std(y) > 1e-12:
        corr = float(np.corrcoef(p, y)[0, 1])
    else:
        corr = None
    return {
        "n": len(p),
        "r": corr,
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "direction": float(np.mean(np.sign(p) == np.sign(y))),
        "mean_prediction": float(np.mean(p)),
        "mean_target": float(np.mean(y)),
    }
