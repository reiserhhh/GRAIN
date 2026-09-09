#!/usr/bin/env python3
"""Explicit odd/even nonlinear distribution contrasts for conditional actions."""

from __future__ import annotations

import numpy as np

ARMS = ("taylor3_bag", "fourier_bag", "taylor3_position", "fourier_position")


def training_geometry(means, variances, training):
    """Use equal-view, equal-author geometry with held views excluded entirely."""
    m, v, keep = (
        np.asarray(means, float),
        np.asarray(variances, float),
        np.asarray(training),
    )
    if (
        m.ndim != 2
        or m.shape != v.shape
        or keep.shape != (len(m),)
        or keep.dtype != bool
        or not keep.any()
        or not np.isfinite(m[keep]).all()
        or not np.isfinite(v[keep]).all()
        or (v[keep] < 0).any()
    ):
        raise ValueError(
            "Finite training view moments and an explicit boolean mask required"
        )
    scale = np.sqrt(v[keep].mean(0))
    return m[keep].mean(0), np.where(scale < 1e-12, 1.0, scale)


def contrast_features(mean, deviation, position, eta, center, scale, arm):
    """Pool conditional Fourier differences or their fixed third-order expansion."""
    c, d, pos, e, m, s = (
        np.asarray(v, float) for v in (mean, deviation, position, eta, center, scale)
    )
    if (
        arm not in ARMS
        or c.ndim != 2
        or c.shape != d.shape
        or not len(c)
        or pos.shape != (len(c),)
        or any(v.shape != (c.shape[1],) for v in (e, m, s))
        or not all(np.isfinite(v).all() for v in (c, d, pos, e, m, s))
        or (s <= 0).any()
    ):
        raise ValueError(
            "Finite token/direction arrays and frozen positive scales required"
        )
    theta, u = (c - m) / s, d / s
    if arm.endswith("position"):
        theta = theta + pos[:, None] * e
    sine = np.sin(u) if arm.startswith("fourier") else u - u**3 / 6
    even = np.cos(u) - 1 if arm.startswith("fourier") else -(u**2) / 2
    norm = np.sqrt(c.shape[1])
    odd_phi = (
        np.concatenate(
            [(-np.sin(theta) * sine).mean(0), (np.cos(theta) * sine).mean(0)]
        )
        / norm
    )
    even_phi = (
        np.concatenate([(np.cos(theta) * even).mean(0), (np.sin(theta) * even).mean(0)])
        / norm
    )
    return odd_phi, even_phi


def fit_action(x, target, penalty=0.01):
    """Zero-intercept, RMS-standardized mean-loss Ridge on endpoint actions."""
    a, y = np.asarray(x, float), np.asarray(target, float)
    if (
        a.ndim != 2
        or y.shape != (len(a),)
        or len(a) < 2
        or not np.isfinite(a).all()
        or not np.isfinite(y).all()
        or not np.isfinite(penalty)
        or penalty <= 0
    ):
        raise ValueError("Aligned finite action features/targets and penalty required")
    scale = np.sqrt(np.mean(a * a, axis=0))
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = a / scale
    coef = np.linalg.solve(
        z.T @ z / len(z) + penalty * np.eye(z.shape[1]), z.T @ y / len(z)
    )
    return {"scale": scale, "coef": coef, "penalty": float(penalty)}


def predict_action(x, model):
    """Evaluate fixed weights; zero contrast gives exactly zero action."""
    a = np.asarray(x, float)
    if a.ndim != 2 or a.shape[1] != len(model["scale"]) or not np.isfinite(a).all():
        raise ValueError("Finite features matching the frozen map required")
    return (a / model["scale"]) @ model["coef"]
