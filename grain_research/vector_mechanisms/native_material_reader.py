#!/usr/bin/env python3
"""Shared material coordinates and token-specific local native readout algebra."""

from __future__ import annotations

import numpy as np


def common_prefix_length(sequences):
    """Count exactly identical leading token IDs across at least two prompts."""
    if len(sequences) < 2 or any(len(s) == 0 for s in sequences):
        raise ValueError("At least two nonempty token sequences required")
    for i in range(min(map(len, sequences))):
        if any(s[i] != sequences[0][i] for s in sequences[1:]):
            return i
    return min(map(len, sequences))


def allocate_slices(offsets, spans, values):
    """Allocate complete body tokens by character overlap with adjacent slices.

    Spans are disjoint, possibly adjacent, absolute character intervals. Tokens
    touching excluded material are not partially assigned. Returned weights sum
    to one for each retained token; no token/text content is needed.
    """
    spans, values = np.asarray(spans, int), np.asarray(values, float)
    if (
        spans.ndim != 2
        or spans.shape[1] != 2
        or values.ndim != 2
        or len(spans) != len(values)
        or len(spans) == 0
        or np.any(spans[:, 1] <= spans[:, 0])
        or np.any(spans[1:, 0] < spans[:-1, 1])
        or not np.isfinite(values).all()
    ):
        raise ValueError("Ordered disjoint slices and finite scores required")
    positions, weights = [], []
    for i, (lo, hi) in enumerate(offsets):
        if hi <= lo:
            continue
        overlap = np.maximum(
            0, np.minimum(hi, spans[:, 1]) - np.maximum(lo, spans[:, 0])
        )
        if int(overlap.sum()) == hi - lo:
            positions.append(i)
            weights.append(overlap / (hi - lo))
    if len(positions) < 2:
        raise ValueError("Insufficient fully covered body tokens")
    weights = np.stack(weights)
    return np.asarray(positions, int), weights @ values, weights


def centered_inputs(h, g):
    """Validate aligned matrices and return their token-centered versions."""
    h, g = np.asarray(h, float), np.asarray(g, float)
    if (
        h.ndim != 2
        or g.ndim != 2
        or len(h) != len(g)
        or len(h) < 2
        or min(h.shape[1], g.shape[1]) == 0
        or not np.isfinite(h).all()
        or not np.isfinite(g).all()
    ):
        raise ValueError("Aligned finite token matrices required")
    return h - h.mean(0), g - g.mean(0)


def direction(h, g, factor):
    """Return one shared GRAIN-modulated contrast in declared relative-RMS units."""
    hc, gc = centered_inputs(h, g)
    if not 0 <= factor < gc.shape[1]:
        raise ValueError("Factor out of range")
    raw = gc[:, factor, None] * hc
    raw_rms = float(np.sqrt(np.mean(raw**2)))
    scale = (
        float(np.sqrt(np.mean(np.asarray(h, float) ** 2))) / raw_rms if raw_rms else 0.0
    )
    return raw * scale, scale


def tangent_parts(h, g, jacobian):
    """Exactly split a local scalar response into common and localized readers.

    The native Jacobian is evaluated at the complete query-specific state, while
    h and g define the common selected material. No finite-change or causal-text
    claim follows merely from this differential identity.
    """
    hc, gc = centered_inputs(h, g)
    jacobian = np.asarray(jacobian, float)
    if jacobian.shape != hc.shape or not np.isfinite(jacobian).all():
        raise ValueError("One finite scalar-output gradient per hidden coordinate")
    c = gc.T @ hc / len(hc)
    mean = jacobian.mean(0)
    common = c @ (len(hc) * mean)
    local = gc.T @ np.sum(hc * (jacobian - mean), axis=1)
    total = gc.T @ np.sum(hc * jacobian, axis=1)
    scales = np.asarray([direction(h, g, k)[1] for k in range(gc.shape[1])])
    return {
        "operator": c,
        "reader": len(hc) * mean,
        "scales": scales,
        "raw_total": total,
        "raw_common": common,
        "raw_local": local,
        "total": total * scales,
        "common": common * scales,
        "local": local * scales,
        "identity_error": float(np.max(abs(total - common - local))),
    }


def cosine(x, y):
    """Return a signed field cosine, or None for a zero field."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Matching finite fields required")
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(x @ y / denominator, -1, 1)) if denominator else None
