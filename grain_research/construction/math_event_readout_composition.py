#!/usr/bin/env python3
"""Finite convex certificates for score-level event aggregation, not model fitting."""

from __future__ import annotations

import numpy as np

REGIMES = ("balanced", "alias_focus", "distinct_focus", "other_focus")
CONTRASTS = (
    ("alias_minus_balanced", "alias_focus", "balanced"),
    ("distinct_minus_alias", "distinct_focus", "alias_focus"),
    ("other_minus_balanced", "other_focus", "balanced"),
)


def vector(value):
    """Require a finite nonempty vector without silently changing its units."""
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Finite nonempty one-dimensional observations required")
    return result


def segment_certificate(value, focal, companion):
    """Project onto an event segment and return an explicit separation witness."""
    z, x, y = (vector(v) for v in (value, focal, companion))
    if z.shape != x.shape or x.shape != y.shape:
        raise ValueError("Matching query coordinates required")
    d = x - y
    norm2 = float(d @ d)
    raw = float((z - y) @ d / norm2) if norm2 > 0 else 0.5
    weight = float(np.clip(raw, 0, 1))
    affine = y + raw * d
    projection = y + weight * d
    residual = z - projection
    lower, upper = np.minimum(x, y), np.maximum(x, y)
    box = np.clip(z, lower, upper)
    gap = float(residual @ z - max(residual @ x, residual @ y))
    return {
        "weight_unconstrained": raw,
        "weight_shared_positive": weight,
        "projection": projection,
        "affine_projection": affine,
        "box_projection": box,
        "witness": residual,
        "segment_distance": float(np.linalg.norm(residual)),
        "affine_distance": float(np.linalg.norm(z - affine)),
        "box_distance": float(np.linalg.norm(z - box)),
        "dual_gap": gap,
        "dual_identity_error": abs(gap - float(residual @ residual)),
        "coordinate_violations": int(np.count_nonzero((z < lower) | (z > upper))),
    }


def update_decomposition(change, focal, companion):
    """Separate an update along a shared event weight from its orthogonal remainder."""
    z, x, y = (vector(v) for v in (change, focal, companion))
    if z.shape != x.shape or x.shape != y.shape:
        raise ValueError("Matching query coordinates required")
    d = x - y
    norm2 = float(d @ d)
    beta = float(z @ d / norm2) if norm2 > 0 else 0.0
    along, remainder = beta * d, z - beta * d
    total = float(z @ z)
    return {
        "shared_weight_change": beta,
        "along": along,
        "remainder": remainder,
        "change_norm": float(np.linalg.norm(z)),
        "remainder_norm": float(np.linalg.norm(remainder)),
        "remainder_energy_share": float(remainder @ remainder / total)
        if total > 0
        else None,
        "orthogonality_error": abs(float(d @ remainder)),
        "pythagorean_error": abs(total - float(along @ along + remainder @ remainder)),
    }


def summarize_certificates(rows, tolerance):
    """Describe finite accounts; do not turn templates into a sampled population."""
    result = {"accounts": len(rows)}
    for name in ("segment", "affine", "box"):
        a = np.array([r[name + "_distance"] for r in rows], float)
        result[name + "_distance_mean"] = float(a.mean())
        result[name + "_distance_rms"] = float(np.sqrt(np.mean(a * a)))
        result[name + "_outside_numerical"] = int(np.count_nonzero(a > tolerance))
    result["inside_box_outside_segment"] = sum(
        r["box_distance"] <= tolerance < r["segment_distance"] for r in rows
    )
    result["maximum_dual_identity_error"] = max(r["dual_identity_error"] for r in rows)
    return result
