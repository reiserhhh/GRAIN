#!/usr/bin/env python3
"""Reference-anchored value/readout candidates and complete response accounting."""

from __future__ import annotations

import numpy as np


def affine_parts(z, h, zref, href, matrix, gradient):
    """Read P and Q deviations separately; never clip extrapolated score units."""
    z, h, zref, href, matrix, gradient = (
        np.asarray(v, dtype=float) for v in (z, h, zref, href, matrix, gradient)
    )
    if z.ndim != 2 or z.shape != zref.shape or z.shape != matrix.shape:
        raise ValueError("P states and matrix must share token/hidden coordinates")
    if h.shape != (z.shape[1],) or href.shape != h.shape or gradient.shape != h.shape:
        raise ValueError("Q state/gradient must share hidden coordinates")
    if not all(np.isfinite(v).all() for v in (z, h, zref, href, matrix, gradient)):
        raise ValueError("Readout inputs must be finite")
    p = float(np.sum(matrix * (z - zref)))
    q = float(gradient @ (h - href))
    return {"P": p, "Q": q, "PQ": p + q}


def response_ledger(full, edited, f00, f10, f01, f11):
    """Account for rebasing, E/P interaction and Q conditional on final E/P."""
    if not np.isfinite([full, edited, f00, f10, f01, f11]).all():
        raise ValueError("Finite cell scores required")
    ep = f11 - f10 - f01 + f00
    return {
        "natural": edited - full,
        "rebase": f00 - full,
        "E_main": f10 - f00,
        "P_main": f01 - f00,
        "EP_interaction": ep,
        "Q_conditional": edited - f11,
        "E_allocated": f10 - f00 + ep / 2,
        "P_allocated": f01 - f00 + ep / 2,
    }


def atlas_readout(z, h, zs, hs, matrices, gradients, scores):
    """Select one fixed chart by joint unscaled P/Q distance, without target access."""
    z, h, zs, hs, matrices, gradients, scores = map(
        np.asarray, (z, h, zs, hs, matrices, gradients, scores)
    )
    distances = np.sum((zs - z) ** 2, axis=(1, 2)) + np.sum((hs - h) ** 2, axis=1)
    index = int(np.argmin(distances))
    parts = affine_parts(z, h, zs[index], hs[index], matrices[index], gradients[index])
    return {"chart": index, "score": float(scores[index] + parts["PQ"])}


def correlation(x, y):
    """Return Pearson correlation for finite, nonconstant vectors, otherwise None."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(x @ y / norm, -1, 1)) if len(x) >= 3 and norm > 1e-14 else None


def paired_quality(actual, predicted, baseline, draws=1000):
    """Pair bootstrap whole authors; report error gains, not per-case effect ratios."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    b = np.broadcast_to(np.asarray(baseline, float), y.shape)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or not len(y)
        or not np.isfinite([y, p, b]).all()
    ):
        raise ValueError("Finite paired author vectors required")
    error, base = (p - y) ** 2, (b - y) ** 2
    rng = np.random.default_rng(20260908)
    gains, rs = [], []
    for _ in range(draws):
        indices = rng.integers(len(y), size=len(y))
        gains.append(float(np.mean(base[indices] - error[indices])))
        r = correlation(y[indices], p[indices])
        if r is not None:
            rs.append(r)
    residual = y - p
    cov = lambda a, c: float(np.mean((a - a.mean()) * (c - c.mean())))
    return {
        "n": len(y),
        "rmse": float(np.sqrt(error.mean())),
        "baseline_rmse": float(np.sqrt(base.mean())),
        "mse_ratio": float(error.mean() / base.mean()) if base.mean() > 1e-16 else None,
        "mse_gain": float((base - error).mean()),
        "mse_gain_ci": np.quantile(gains, [0.025, 0.975]).tolist(),
        "r": correlation(y, p),
        "r_ci": np.quantile(rs, [0.025, 0.975]).tolist() if rs else [None, None],
        "same_sign_centered": int(np.sum((p - b) * (y - b) > 0)),
        "var_actual": cov(y, y),
        "var_predicted": cov(p, p),
        "var_remainder": cov(residual, residual),
        "cov_predicted_remainder": cov(p, residual),
        "cov_predicted_actual": cov(p, y),
        "cov_remainder_actual": cov(residual, y),
    }
