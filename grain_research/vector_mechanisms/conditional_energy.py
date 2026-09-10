#!/usr/bin/env python3
"""Positive finite conditional-action mixtures and exact first/second moments."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.comment_carrier import score
from grain_research.vector_mechanisms.joint_readout import pair_kernel


def fit_distribution(left, right, targets, scale, penalty=0.01):
    """Store source-only support and a fixed zero-atom mass, without optimization."""
    y = np.asarray(targets, float)
    k = pair_kernel(left, right, left, right)
    if y.ndim != 2 or len(y) != len(k) or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Finite paired training action vectors required")
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Positive frozen scale and penalty required")
    alpha = penalty * float(np.mean(np.sum((k / scale) ** 2, axis=1)))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Training pair support must have positive finite energy")
    return {
        "left": np.array(left, dtype=float, copy=True),
        "right": np.array(right, dtype=float, copy=True),
        "targets": y.copy(),
        "scale": float(scale),
        "alpha": alpha,
        "penalty": float(penalty),
    }


def mixture_weights(left, right, model):
    """Return weights including the first, zero-action atom; preserve exact zero."""
    cross = pair_kernel(left, right, model["left"], model["right"])
    h = (cross / model["scale"]) ** 2
    weights = np.column_stack([np.full(len(h), model["alpha"]), h])
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Nonnegative finite kernel mass required")
    return weights / weights.sum(axis=1, keepdims=True)


def mixture_moments(weights, targets):
    """Compute moments of the actual finite mixture, not a calibrated posterior."""
    w, y = np.asarray(weights, float), np.asarray(targets, float)
    if (
        w.ndim != 2
        or y.ndim != 2
        or w.shape[1] != len(y) + 1
        or not np.isfinite(w).all()
        or not np.isfinite(y).all()
        or (w < 0).any()
        or not np.allclose(w.sum(1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Normalized nonnegative weights including zero atom required")
    atoms = np.vstack([np.zeros((1, y.shape[1])), y])
    mean = w @ atoms
    second = np.einsum("ni,ij,ik->njk", w, atoms, atoms)
    covariance = second - np.einsum("ni,nj->nij", mean, mean)
    energy = w @ np.sum(atoms**2, axis=1)
    mean_energy = np.sum(mean**2, axis=1)
    return {
        "mean": mean,
        "second": second,
        "covariance": covariance,
        "energy": energy,
        "mean_energy": mean_energy,
        "variance_trace": np.trace(covariance, axis1=1, axis2=2),
        "reference_mass": 1 - w[:, 0],
        "effective_atoms": 1 / np.sum(w**2, axis=1),
    }


def energy_gap(actual, predicted):
    """Keep the finite OOF cross term rather than call any energy deficit variance."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.ndim != 2 or y.shape != p.shape or not np.isfinite([y, p]).all():
        raise ValueError("Matched finite action vectors required")
    residual = y - p
    gap = np.sum(y**2 - p**2, axis=1)
    error = np.sum(residual**2, axis=1)
    cross = 2 * np.sum(p * residual, axis=1)
    return {
        "energy_gap": gap,
        "residual_energy": error,
        "cross_term": cross,
        "identity_error": abs(gap - error - cross),
    }


def score_hessian(logits, values):
    """Analytic Hessian of the original alphabet-softmax numeric expectation."""
    z, values = np.asarray(logits, float), np.asarray(values, float)
    if z.ndim != 1 or z.shape != values.shape or not np.isfinite([z, values]).all():
        raise ValueError("Matched finite candidate logits and values required")
    p = np.exp(z - z.max())
    p /= p.sum()
    g = p * (values - p @ values)
    return np.diag(g) - np.outer(g, p) - np.outer(p, g)


def score_mixture(origin, values, weights, targets):
    """Account for exact mixture versus mean-score and a covariance correction."""
    w = np.asarray(weights, float)
    if w.ndim != 1:
        raise ValueError("One mixture row required")
    moments = mixture_moments(w[None], targets)
    atoms = np.vstack([np.zeros((1, len(origin))), targets])
    mean = moments["mean"][0]
    at_mean = score(np.asarray(origin) + mean, values)
    expected = float(
        w @ np.array([score(np.asarray(origin) + atom, values) for atom in atoms])
    )
    correction = float(
        np.sum(
            score_hessian(np.asarray(origin) + mean, values) * moments["covariance"][0]
        )
        / 2
    )
    return {
        "score_at_mean": at_mean,
        "mean_score": expected,
        "mixture_minus_mean_score": expected - at_mean,
        "covariance_correction": correction,
        "higher_order_remainder": expected - at_mean - correction,
    }
