#!/usr/bin/env python3
"""Symmetric tensor readouts for measured mixed candidate-logit actions."""

from __future__ import annotations

import numpy as np


def pair_kernel(a, b, c, d):
    """Kernel of (a tensor b + b tensor a)/2, preserving swap and zero properties."""
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
    if (
        any(x.ndim != 2 or not np.isfinite(x).all() for x in (a, b, c, d))
        or a.shape != b.shape
        or c.shape != d.shape
        or a.shape[1] != c.shape[1]
    ):
        raise ValueError("Matched finite comment-feature pairs required")
    return ((a @ c.T) * (b @ d.T) + (a @ d.T) * (b @ c.T)) / 2


def fit_joint(a, b, targets, penalty=0.01):
    """Train mean-loss zero-intercept kernel Ridge using training normalization only."""
    y = np.asarray(targets, float)
    k = pair_kernel(a, b, a, b)
    if (
        y.ndim != 2
        or len(y) != len(k)
        or len(k) < 2
        or not np.isfinite(y).all()
        or not np.isfinite(penalty)
        or penalty <= 0
    ):
        raise ValueError("Finite paired logit targets and positive penalty required")
    scale = float(np.trace(k) / len(k))
    if scale <= 1e-20:
        raise ValueError("Training pair features have no energy")
    weights = np.linalg.solve(k / scale + len(k) * penalty * np.eye(len(k)), y)
    return {
        "left": np.asarray(a),
        "right": np.asarray(b),
        "scale": scale,
        "weights": weights,
        "penalty": float(penalty),
    }


def predict_joint(a, b, model):
    """Read new paired features without evaluation labels or normalization fitting."""
    return (
        pair_kernel(a, b, model["left"], model["right"])
        @ model["weights"]
        / model["scale"]
    )


def centered_logits(z):
    """Remove the common answer-logit shift, which cannot affect conditional scores."""
    a = np.asarray(z, float)
    if a.ndim < 1 or not np.isfinite(a).all():
        raise ValueError("Finite candidate logits required")
    return a - a.mean(axis=-1, keepdims=True)


def vector_quality(actual, predicted):
    """Report full-vector fidelity without treating answer classes as independent people."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.shape != p.shape or y.ndim != 2 or not np.isfinite([y, p]).all():
        raise ValueError("Finite aligned action matrices required")
    from grain_research.vector_mechanisms.level_response import correlation

    return {
        "mse_over_zero": float(np.sum((y - p) ** 2) / np.sum(y * y)),
        "norm_r": correlation(np.linalg.norm(y, axis=1), np.linalg.norm(p, axis=1)),
        "centered_vector_r": correlation(y.ravel(), p.ravel()),
        "actual_rms": float(np.sqrt(np.mean(y * y))),
        "predicted_rms": float(np.sqrt(np.mean(p * p))),
    }
