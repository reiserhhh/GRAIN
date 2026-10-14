#!/usr/bin/env python3
"""HM27 common covectors on variable token-moment observation quotients."""

from __future__ import annotations

import numpy as np


def moment_map(tokens):
    """Return the fixed four-moment observation for one to four tokens."""
    if not isinstance(tokens, int) or not 1 <= tokens <= 4:
        raise ValueError("this finite moment bank requires1-4 tokens")
    t = np.zeros(1) if tokens == 1 else np.linspace(-1, 1, tokens)
    return np.array([t**k / tokens for k in range(4)])


def svd_inverse(a):
    """Return a machine-threshold inverse/support without an outcome-selected rank."""
    a = np.asarray(a, float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("finite matrix required")
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    cutoff = 100 * np.finfo(float).eps * max(a.shape) * s[0]
    rank = int(sum(s > cutoff))
    if not rank:
        raise ValueError("observation is zero")
    inverse = (vt[:rank].T / s[:rank]) @ u[:, :rank].T
    return inverse, vt[:rank].T @ vt[:rank], s, rank


def extension(gradient, frame=None):
    """Select a minimum-norm moment extension, explicitly retaining its null space."""
    g = np.asarray(gradient, float)
    if g.ndim != 2 or not np.isfinite(g).all():
        raise ValueError("finite token-by-hidden gradient required")
    frame = np.eye(4) if frame is None else np.asarray(frame, float)
    if (
        frame.shape != (4, 4)
        or not np.isfinite(frame).all()
        or np.linalg.matrix_rank(frame) != 4
    ):
        raise ValueError("invertible fixed moment frame required")
    o = frame @ moment_map(len(g))
    inverse, support, singular, rank = svd_inverse(o.T)
    covector = inverse @ g
    return {
        "covector": covector,
        "support": support,
        "singular": singular,
        "rank": rank,
        "residual_max": float(np.max(abs(o.T @ covector - g))),
    }


def common_covector(gradients, frame=None):
    """Jointly solve equal-family token-mean constraints, not averages of extensions."""
    if not gradients:
        raise ValueError("source gradient families required")
    frame = np.eye(4) if frame is None else np.asarray(frame, float)
    if (
        frame.shape != (4, 4)
        or not np.isfinite(frame).all()
        or np.linalg.matrix_rank(frame) != 4
    ):
        raise ValueError("invertible moment frame required")
    gs = [np.asarray(g, float) for g in gradients]
    if any(
        g.ndim != 2 or not np.isfinite(g).all() or g.shape[1] != gs[0].shape[1]
        for g in gs
    ):
        raise ValueError("matched finite source gradients required")
    weights = [1 / np.sqrt(len(g) * len(gs)) for g in gs]
    a = np.vstack(
        [(frame @ moment_map(len(g))).T * w for g, w in zip(gs, weights, strict=True)]
    )
    b = np.vstack([g * w for g, w in zip(gs, weights, strict=True)])
    inverse, support, singular, rank = svd_inverse(a)
    covector = inverse @ b
    residual = a @ covector - b
    return {
        "covector": covector,
        "support": support,
        "singular": singular,
        "rank": rank,
        "weighted_residual_rms": float(np.sqrt(np.sum(residual**2) / b.shape[1])),
    }


def contract(covector, displacement):
    """Contract a moment covector with a same-frame displacement in physical units."""
    a, x = np.asarray(covector, float), np.asarray(displacement, float)
    if a.shape != x.shape or not np.isfinite(a).all() or not np.isfinite(x).all():
        raise ValueError("matched finite covector and displacement required")
    return float(np.sum(a * x))


def paired_covariance(native_left, native_right, prediction_left, prediction_right):
    """Keep signed PP/PR/RP/RR terms rather than inventing variance percentages."""
    y, z, a, b = map(
        np.asarray, (native_left, native_right, prediction_left, prediction_right)
    )
    if (
        y.ndim != 1
        or len(y) < 2
        or any(v.shape != y.shape or not np.isfinite(v).all() for v in (y, z, a, b))
    ):
        raise ValueError("finite family-matched paired queries required")
    cov = lambda x, w: float(np.mean((x - x.mean()) * (w - w.mean())))
    return {
        "native": cov(y, z),
        "PP": cov(a, b),
        "PR": cov(a, z - b),
        "RP": cov(y - a, b),
        "RR": cov(y - a, z - b),
    }
