#!/usr/bin/env python3
"""Lossless target-conditioning accounts on complete ownership evidence cubes."""

from __future__ import annotations

import itertools

import numpy as np

TARGETS = (
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
    "anthropomorphism",
)
CHARACTERS = ("constant", "g", "y", "yg", "x", "xg", "xy", "xyg")
POINTS = np.asarray(list(itertools.product((-1, 1), repeat=3)), dtype=float)
SIGNS = np.asarray(
    [
        np.prod(POINTS[:, [j for j in range(3) if mask & (1 << (2 - j))]], axis=1)
        for mask in range(8)
    ]
)


def complete(values):
    """Require all eight corners and six matched queries in explicit axes."""
    a = np.asarray(values, dtype=float)
    if a.ndim < 2 or a.shape[:2] != (8, 6) or not np.isfinite(a).all():
        raise ValueError("Complete finite 8-cell by6-query observations required")
    return a


def coefficients(values):
    """Retain every Boolean coefficient, including the reference level."""
    a = complete(values)
    return np.tensordot(SIGNS / 8, a, axes=(1, 0))


def query_account(value):
    """Split all queries losslessly into shared and zero-sum varying actions."""
    a = np.asarray(value, dtype=float)
    if a.ndim < 1 or a.shape[0] != 6 or not np.isfinite(a).all():
        raise ValueError("Six finite matched query actions required")
    shared = a.mean(axis=0)
    varying = a - shared
    shared_energy = float(6 * np.square(shared).sum())
    varying_energy = float(np.square(varying).sum())
    total = float(np.square(a).sum())
    return {
        "shared": shared,
        "varying": varying,
        "shared_energy": shared_energy,
        "varying_energy": varying_energy,
        "total_energy": total,
        "varying_fraction": varying_energy / total if total > 1e-24 else None,
        "energy_error": abs(total - shared_energy - varying_energy),
        "reconstruction_error": float(np.max(abs(a - shared - varying))),
        "query_sum_error": float(np.max(abs(varying.sum(axis=0)))),
    }


def ownership_action(coef, focal=None):
    """Return the exact companion polarity-by-owner difference, keeping context."""
    c = complete(coef)
    if focal not in (None, -1, 1):
        raise ValueError("Focal condition must be None, -1 or +1")
    return 4 * (c[3] if focal is None else c[3] + focal * c[7])


def direct_ownership_action(values, focal=None):
    """Evaluate the same four-corner action without Fourier arithmetic."""
    if focal not in (None, -1, 1):
        raise ValueError("Focal condition must be None, -1 or +1")
    a = complete(values).reshape(2, 2, 2, *np.shape(values)[1:])
    effects = a[:, 1, 1] - a[:, 0, 1] - a[:, 1, 0] + a[:, 0, 0]
    return effects.mean(axis=0) if focal is None else effects[int(focal > 0)]


def cosine(a, b):
    """Return a shape-preserving direction comparison, with explicit nulls."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Matched finite actions required")
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-12:
        return None
    return float(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def matched_contrast(action, companion):
    """Describe named versus other query responses without a significance claim."""
    a = np.asarray(action, float)
    if a.shape != (6,) or not np.isfinite(a).all() or companion not in TARGETS:
        raise ValueError("Six scalar query actions and a known companion required")
    k = TARGETS.index(companion)
    others = np.delete(a, k)
    return {
        "matched": float(a[k]),
        "other_mean": float(others.mean()),
        "matched_minus_other": float(a[k] - others.mean()),
        "matched_is_max": bool(a[k] >= a.max() - 1e-12),
    }


def toy_world(kind):
    """Build deterministic alternatives that prevent generic changes passing as binding."""
    x, y, g = (POINTS[:, j, None] for j in range(3))
    label = np.eye(6)[2][None, :]
    if kind == "generic":
        return np.broadcast_to(0.2 * y * g, (8, 6)).copy()
    if kind == "selective":
        return 0.2 * y * g * label
    if kind == "conditional":
        return 0.2 * x * y * g * label
    if kind == "role_only":
        return np.broadcast_to(0.2 * g, (8, 6)).copy()
    raise ValueError("Unknown mechanism world")
