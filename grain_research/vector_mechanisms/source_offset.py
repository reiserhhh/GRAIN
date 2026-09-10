#!/usr/bin/env python3
"""Exact source-intercept ingredients and complete-function factorial accounts."""

from __future__ import annotations

import numpy as np

from .response_repair import allocate, summarize
from .source_state_transport import apply_map

CANDIDATES = ("identity", "offset_only", "nonintercept_only", "joint")
NEW_CANDIDATES = CANDIDATES[1:3]
RENAMES = {
    "state": "offset",
    "tangent": "remainder",
    "value": "identity",
    "joint": "full",
}


def family(x, d, model):
    """Split the explicit source intercept, not a current-material mean or target."""
    x, d = np.asarray(x, float), np.asarray(d, float)
    if str(model["mode"]) != "pointwise":
        raise ValueError("Frozen pointwise source function required")
    bias = model["coef"].shape[0] - 3
    if bias != model["basis"].shape[1] or (
        model["center"][bias] != 0 or model["scale"][bias] != 1
    ):
        raise ValueError("Explicit uncentered unit source-intercept feature required")
    mx, md, z, dz = apply_map(x, d, model)
    if not np.all(z[:, bias] == 1) or not np.all(dz[:, :, bias] == 0):
        raise ValueError("Source intercept or layout derivative changed")
    if not all(np.isfinite(v).all() for v in (mx, md, z, dz)):
        raise ValueError("Finite source function required")
    b = model["coef"][bias]
    offset, remainder = x + b, mx - b
    np.testing.assert_allclose(mx, offset + remainder - x, atol=1e-11, rtol=1e-12)
    return {
        "identity": (x, d),
        "offset_only": (offset, d),
        "nonintercept_only": (remainder, md),
        "joint": (mx, md),
    }


def offset_geometry(x, model, mapped):
    """Account offset/remainder energy with its signed cross term, not percentages."""
    b = model["coef"][model["coef"].shape[0] - 3]
    delta = mapped - x
    remainder = delta - b
    total = float(np.mean(delta**2))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Nonzero finite correction energy required")
    eb, ev = float(np.mean(b**2)), float(np.mean(remainder**2))
    cross = float(2 * np.mean(b * remainder))
    mu = delta.mean(0)
    return {
        "correction_energy": total,
        "offset_energy": eb,
        "remainder_energy": ev,
        "signed_cross_energy": cross,
        "offset_over_total_energy": eb / total,
        "remainder_over_total_energy": ev / total,
        "signed_cross_over_total_energy": cross / total,
        "energy_identity_error": abs(total - eb - ev - cross),
        "offset_vs_material_mean_relative_error": float(
            np.linalg.norm(b - mu) / np.linalg.norm(mu)
        ),
        "offset_vs_material_mean_cosine": float(
            np.dot(b, mu) / (np.linalg.norm(b) * np.linalg.norm(mu))
        ),
    }


def _rename(record, reverse=False):
    pairs = {v: k for k, v in RENAMES.items()} if reverse else RENAMES
    result = {}
    for key, value in record.items():
        for old, new in pairs.items():
            key = key.replace(old, new)
        result[key] = value
    return result


def ingredient_effects(values, actual):
    """Order I/O/V/F into offset-by-remainder factorial cells and signed errors."""
    a = np.asarray(values, float)
    if a.shape[0] != 4:
        raise ValueError("I/O/V/F inventory required")
    cells = np.stack([a[0], a[2], a[1], a[3]]).reshape((2, 2) + a.shape[1:])
    return _rename(allocate(cells, actual))


def effect_summary(columns):
    """Aggregate exact ingredient changes; preserve cancellation and interactions."""
    return _rename(summarize(_rename(columns, reverse=True)))
