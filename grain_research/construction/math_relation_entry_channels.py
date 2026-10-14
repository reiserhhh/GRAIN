#!/usr/bin/env python3
"""Finite conditional-relation increments and recursive reader entry accounts."""

from __future__ import annotations

import numpy as np


def relation_increment(cube, e, n):
    """Return the exact mixed source increment without an output-trained probe."""
    x = np.asarray(cube, dtype=np.float64)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete finite source cube required")
    if e not in (-1, 1) or n not in (-1, 1):
        raise ValueError("Invalid actor signs")
    mixed = (x[0, 0] - x[0, 1] - x[1, 0] + x[1, 1]) / 4
    return -2 * e * n * mixed


def entry_components(base, key, value, joint):
    """Account for all four recursive-reader corners without mediation claims."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (base, key, value, joint)]
    if len({x.shape for x in arrays}) != 1 or any(
        not np.isfinite(x).all() for x in arrays
    ):
        raise ValueError("Aligned finite reader observations required")
    b, k, v, j = arrays
    return {
        "key_entry": k - b,
        "value_entry": v - b,
        "recursive_interaction": j - k - v + b,
        "joint_entry": j - b,
    }


def action_description(x, reference):
    """Describe signed action, preserving undefined null-reference comparisons."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or not x.size or not np.isfinite([x, y]).all():
        raise ValueError("Aligned nonempty finite action vectors required")
    xx, yy, xy = float(x @ x), float(y @ y), float(x @ y)
    return {
        "count": int(x.size),
        "rms": float(np.sqrt(xx / x.size)),
        "reference_rms": float(np.sqrt(yy / x.size)),
        "same_reference_direction": int(np.sum(x * y > 0)),
        "signed_projection": xy / yy if yy else None,
        "relative_difference": float(np.linalg.norm(x - y) / np.sqrt(yy))
        if yy
        else None,
    }
