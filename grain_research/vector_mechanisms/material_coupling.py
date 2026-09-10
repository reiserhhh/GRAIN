#!/usr/bin/env python3
"""Same-material expression coordinates and fixed-query functional co-location."""

from __future__ import annotations

import numpy as np


def local_coupling(grain, functional):
    """Measure within-material alignment without interpreting weights as attention."""
    g, p = np.asarray(grain, float), np.asarray(functional, float)
    if g.ndim != 2 or p.shape != (len(g),) or len(g) < 2:
        raise ValueError("At least two aligned comments and a feature matrix required")
    if not np.isfinite(g).all() or not np.isfinite(p).all():
        raise ValueError("Finite observations required")
    u, pc = g.mean(0), p - p.mean()
    gc = g - u
    a = np.abs(pc)
    weights = a / a.sum() if a.sum() > 1e-14 else np.full(len(g), 1 / len(g))
    weighted = weights @ g
    covariance_identity = (
        np.mean(gc * (a - a.mean())[:, None], axis=0) / a.mean()
        if a.sum() > 1e-14
        else np.zeros(g.shape[1])
    )
    return {
        "U": u,
        "W": weighted,
        "D": weighted - u,
        "S": np.mean(gc * pc[:, None], axis=0),
        "shifted_W": np.roll(weights, 1) @ g,
        "g_variance": np.mean(gc**2, axis=0),
        "p_variance": float(np.mean(pc**2)),
        "effective_comments": float(1 / np.sum(weights**2)),
        "eligible_comments": len(g),
        "identity_error": float(np.max(abs(weighted - u - covariance_identity))),
    }


def row_correlations(a, b):
    """Compute correlations for matched bootstrap rows; constants stay undefined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError("Finite aligned bootstrap matrices required")
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    denom = np.sqrt(np.sum(ac**2, axis=1) * np.sum(bc**2, axis=1))
    return np.divide(
        np.sum(ac * bc, axis=1),
        denom,
        out=np.full(len(a), np.nan),
        where=denom > 1e-14,
    )


def span_allocation(token_spans, slice_spans):
    """Allocate character-addressed token effects to disjoint exhaustive slices.

    Crossing-token effects are split by overlap length as an accounting rule,
    not independently identified within-token causal effects.
    """
    tokens, slices = np.asarray(token_spans, int), np.asarray(slice_spans, int)
    for spans in (tokens, slices):
        if spans.ndim != 2 or spans.shape[1] != 2 or len(spans) == 0:
            raise ValueError("Nonempty start/end arrays required")
        if np.any(spans[:, 0] < 0) or np.any(spans[:, 1] <= spans[:, 0]):
            raise ValueError("Nonempty nonnegative character intervals required")
    if np.any(slices[1:, 0] != slices[:-1, 1]):
        raise ValueError("Slices must be contiguous, disjoint and ordered")
    overlap = np.maximum(
        np.minimum(slices[:, None, 1], tokens[None, :, 1])
        - np.maximum(slices[:, None, 0], tokens[None, :, 0]),
        0,
    )
    weights = overlap / (tokens[:, 1] - tokens[:, 0])[None]
    if not np.allclose(weights.sum(0), 1, rtol=0, atol=1e-12):
        raise ValueError("Every supplied token must be covered by the slice domain")
    return weights
