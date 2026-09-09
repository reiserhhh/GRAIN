#!/usr/bin/env python3
"""Compile a centered linear-kernel state emulator into one affine early-state reader."""

from __future__ import annotations

import numpy as np

from .state_geometry import linear_reader


def compile_affine(model, function):
    """Eliminate kernel anchors and full predicted states without changing the score."""
    reader, bias = linear_reader(function)
    u = np.asarray(model["train_x"], float)
    wl = np.asarray(model["coef1"], float) @ reader
    correction = (u - u.mean(0)).T @ wl / float(model["scale"])
    coefficients = reader + correction
    offset = (
        bias
        + model["mean_r"] @ reader
        - model["mean_x"] @ correction
        + (-model["col1"] + model["grand1"]) @ wl
    )
    if not np.isfinite(coefficients).all() or not np.isfinite(offset).all():
        raise ValueError("Nonfinite affine reader")
    return {
        "coefficients": coefficients,
        "offset": offset,
        "anchor": np.asarray(function.anchor),
    }


def score_affine(program, early):
    """Return three unchanged contributions and an unclipped score."""
    x = np.asarray(early, float)
    c, b = np.asarray(program["coefficients"]), np.asarray(program["offset"])
    if (
        x.ndim != 1
        or c.shape != (len(x), 3)
        or b.shape != (3,)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Affine reader coordinate mismatch")
    parts = x @ c + b
    return {
        "mean": float(parts[0]),
        "configuration": float(parts[1]),
        "final": float(parts[2]),
        "score": float(program["anchor"] + parts.sum()),
    }
