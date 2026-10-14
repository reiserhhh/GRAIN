#!/usr/bin/env python3
"""HM24: differential observation fibers in typed token-moment coordinates."""

from __future__ import annotations

import numpy as np


def observation_matrix(tokens, degree):
    """Return the declared moment map, without choosing degree from outcomes."""
    if not isinstance(tokens, int) or tokens < 1 or degree not in (0, 3):
        raise ValueError("positive token count and fixed degree0/3 required")
    p = np.zeros(1) if tokens == 1 else np.linspace(-1, 1, tokens)
    return np.array([p**k / tokens for k in range(degree + 1)])


def split_gradient(gradient, degree):
    """Split a covector into visible and moment-null temporal components."""
    g = np.asarray(gradient, float)
    if g.ndim != 2 or min(g.shape) < 1 or not np.isfinite(g).all():
        raise ValueError("finite token-by-hidden gradient required")
    o = observation_matrix(len(g), degree)
    u, singular, vt = np.linalg.svd(o, full_matrices=False)
    cutoff = 100 * np.finfo(float).eps * max(o.shape) * singular[0]
    rank = int(sum(singular > cutoff))
    v = vt[:rank].T
    visible = g.copy() if rank == len(g) else v @ (v.T @ g)
    null = np.zeros_like(g) if rank == len(g) else g - visible
    covector = (u[:, :rank] / singular[:rank]) @ (v.T @ g)
    return {
        "visible": visible,
        "null": null,
        "moment_covector": covector,
        "rank": rank,
        "tokens": len(g),
        "null_tokens": len(g) - rank,
        "null_observation_error": float(np.max(abs(o @ null), initial=0)),
        "reconstruction_error": float(np.max(abs(visible + null - g), initial=0)),
        "covector_error": float(np.max(abs(o.T @ covector - visible), initial=0)),
    }


def normalize_direction(blocks, radius):
    """Normalize a jointly addressed direction in Frobenius units, preserving zeros."""
    if not blocks or not np.isfinite(radius) or radius <= 0:
        raise ValueError("positive fixed radius and blocks required")
    arrays = {k: np.asarray(v, float) for k, v in blocks.items()}
    if any(v.ndim != 2 or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("finite token-by-hidden blocks required")
    norm = np.sqrt(sum(float(np.sum(v**2)) for v in arrays.values()))
    return {
        k: v * (radius / norm) if norm else np.zeros_like(v) for k, v in arrays.items()
    }


def directional_action(gradients, deltas):
    """Contract matched actual-state covectors and directions in physical units."""
    if set(gradients) != set(deltas):
        raise ValueError("matched typed spans required")
    result = 0.0
    for k in gradients:
        g, d = np.asarray(gradients[k], float), np.asarray(deltas[k], float)
        if g.shape != d.shape or not np.isfinite(g).all() or not np.isfinite(d).all():
            raise ValueError("matched finite directions required")
        result += float(np.sum(g * d))
    return result
