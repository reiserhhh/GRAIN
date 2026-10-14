#!/usr/bin/env python3
"""Exact function-class bounds for addressed categorical context actions."""

from __future__ import annotations

import numpy as np

RESOLUTION = 2e-12


def validate_measure(measure):
    """Require finite zero-mass signed measures; never silently recenter."""
    m = np.asarray(measure, float)
    if m.ndim == 0 or m.shape[-1] < 2 or not np.isfinite(m).all():
        raise ValueError("Finite ordered alphabet with at least two entries required")
    if np.any(np.abs(m.sum(-1)) > RESOLUTION):
        raise ValueError("The action is not a zero-mass signed measure")
    return m


def tail_actions(measure):
    """Generators of the normalized increasing-valuation cone,not fitted weights."""
    m = validate_measure(measure)
    return np.cumsum(m[..., ::-1], axis=-1)[..., ::-1][..., 1:]


def readout(measure, valuation):
    """Evaluate a declared increasing scale,without selecting it from outcomes."""
    m, v = validate_measure(measure), np.asarray(valuation, float)
    if (
        v.shape != (m.shape[-1],)
        or not np.isfinite(v).all()
        or v[0] != 0
        or v[-1] != 1
        or np.any(np.diff(v) < 0)
    ):
        raise ValueError("Normalized increasing valuation required")
    return m @ v


def certificate(measure):
    """Exact attainable bounds for weak order;strict-order infimum/supremum."""
    m = validate_measure(measure)
    if m.ndim != 1:
        raise ValueError("One signed measure per certificate required")
    t = tail_actions(m)
    lo, hi = float(t.min()), float(t.max())
    positive = int(np.count_nonzero(t > RESOLUTION))
    negative = int(np.count_nonzero(t < -RESOLUTION))
    if positive and negative:
        status = "SCALE_DEPENDENT"
    elif positive:
        status = "ORDINALLY_NONNEGATIVE"
    elif negative:
        status = "ORDINALLY_NONPOSITIVE"
    else:
        status = "NUMERICALLY_UNRESOLVED_ZERO"
    return {
        "lower": lo,
        "upper": hi,
        "equally_spaced": float(m @ np.linspace(0, 1, len(m))),
        "tail_mean": float(t.mean()),
        "positive_generators": positive,
        "negative_generators": negative,
        "unresolved_generators": len(t) - positive - negative,
        "all_generators_strictly_positive": bool(lo > RESOLUTION),
        "all_generators_strictly_negative": bool(hi < -RESOLUTION),
        "status": status,
        "mass_error": float(abs(m.sum())),
        "resolution": RESOLUTION,
    }


def source_alignment(source, native_measure):
    """Source-normalized linear pairing;zero source is not an identified target."""
    x, m = np.asarray(source, float), validate_measure(native_measure)
    if x.ndim != 1 or m.ndim != 2 or len(x) != len(m) or not np.isfinite(x).all():
        raise ValueError("SAME identified finite author array required")
    norm = float(np.linalg.norm(x))
    if norm <= RESOLUTION:
        return None, {"status": "UNRESOLVED_SOURCE_ZERO", "source_norm": norm}
    b = (x / norm) @ m
    return b, certificate(b) | {"source_norm": norm, "authors": len(x)}


def native_action(logits):
    """Five SAME natural subsets to a conditional digit-distribution action."""
    x = np.asarray(logits, float)
    if x.ndim < 2 or x.shape[-2:] != (5, 10) or not np.isfinite(x).all():
        raise ValueError("Five natural subsets and ten cached digit logits required")
    exp = np.exp(x - x.max(-1, keepdims=True))
    p = exp / exp.sum(-1, keepdims=True)
    m = p[..., 2, :] - p[..., 1, :] - p[..., 3, :] + p[..., 4, :]
    positive = (p[..., 2, :] + p[..., 4, :]) / 2
    negative = (p[..., 1, :] + p[..., 3, :]) / 2
    validate_measure(m)
    return p, m, positive, negative


def oriented_individual(source, measure):
    """Keep individual SOURCE direction;do not label its unresolved zero a null."""
    x, m = float(source), validate_measure(measure)
    if not np.isfinite(x) or m.ndim != 1:
        raise ValueError("One finite source value and signed measure required")
    if abs(x) <= RESOLUTION:
        return {"status": "UNRESOLVED_SOURCE_ZERO", "source_value": x}
    return certificate(np.sign(x) * m) | {"source_value": x}
