#!/usr/bin/env python3
"""Audit available native interventions without promoting an incomplete battery."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_joint_reading import BASIS


def complete_groups(records, groups, targets, arms):
    """Require every actual corner, target and arm; missing data are never zero."""
    return [
        g
        for g in groups
        if all(
            (g, arm, corner, target) in records
            for arm in arms
            for corner in range(4)
            for target in range(targets)
        )
    ]


def character_table(corners):
    """Compute all four character coefficients and independently check inversion."""
    x = np.asarray(corners, float)
    if x.ndim < 2 or x.shape[0] != 4 or not np.isfinite(x).all():
        raise ValueError("Four complete finite corners required")
    chars = np.tensordot(BASIS / 4, x, axes=1)
    rebuilt = np.stack(
        [sum(BASIS[j, c] * chars[j] for j in range(4)) for c in range(4)]
    )
    if not np.allclose(rebuilt, x, rtol=0, atol=1e-9):
        raise ValueError("Character inversion failed")
    return chars


def matrix_sectors(table):
    """Retain global, action, target and coupled components of a complete table."""
    x = np.asarray(table, float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("Complete action-by-target matrix required")
    global_part = np.full_like(x, x.mean())
    action = np.broadcast_to(x.mean(axis=1, keepdims=True), x.shape) - global_part
    target = np.broadcast_to(x.mean(axis=0, keepdims=True), x.shape) - global_part
    coupled = x - global_part - action - target
    parts = {
        "global_part": global_part,
        "action": action,
        "target": target,
        "coupled": coupled,
    }
    total = sum(float(np.vdot(v, v)) for v in parts.values())
    if not np.allclose(total, np.vdot(x, x), rtol=1e-10, atol=1e-10):
        raise ValueError("Complete sector energy failed")
    return parts


def cosine(x, y):
    """Return signed geometric agreement, not psychological correlation."""
    a, b = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Aligned finite arrays required")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else None


def describe(x):
    """Keep magnitude and sign without selecting a winning target or row."""
    a = np.asarray(x, float).ravel()
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Nonempty finite observations required")
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "rms": float(np.sqrt(a @ a / a.size)),
        "minimum": float(a.min()),
        "maximum": float(a.max()),
    }


def effect_summary(full, removed, random, local=None):
    """Compare actual directed effects, random controls and residual native signal."""
    full, removed, random = [np.asarray(v, float) for v in (full, removed, random)]
    if not (full.shape == removed.shape == random.shape):
        raise ValueError("Matched complete observations required")
    denom = float(np.vdot(full, full))
    result = {
        "native": describe(full),
        "removed_effect": describe(removed),
        "random_effect": describe(random),
        "remaining_native": describe(full - removed),
        "removed_alignment": float(np.vdot(removed, full) / denom) if denom else None,
        "random_alignment": float(np.vdot(random, full) / denom) if denom else None,
        "removed_native_cosine": cosine(removed, full),
        "random_native_cosine": cosine(random, full),
    }
    if local is not None:
        result["native_local_cosine"] = cosine(full, local)
    return result
