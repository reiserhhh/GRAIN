#!/usr/bin/env python3
"""Split source-fitted role observations into common and target-specific maps."""

from __future__ import annotations

import numpy as np


def affine_roles(model):
    """Compile p/o/a in the common declared units, preserving the intercept."""
    w, mu, bias = (np.asarray(model[k], dtype=float) for k in ("w", "mu", "bias"))
    if (
        w.ndim != 2
        or w.shape[1] != 4
        or mu.shape != (w.shape[0],)
        or bias.shape != (4,)
    ):
        raise ValueError("frozen four-coordinate source program required")
    return {"w": w[:, 1:].copy(), "intercept": (bias - mu @ w)[1:]}


def split_common_roles(left, right):
    """Use a prespecified equal common map; no application-selected weights."""
    a, b = affine_roles(left), affine_roles(right)
    if a["w"].shape != b["w"].shape:
        raise ValueError("shared roles require identical reader coordinates")
    common = {key: (a[key] + b[key]) / 2 for key in a}
    difference = {key: (a[key] - b[key]) / 2 for key in a}
    return common, difference


def score_roles(program, x):
    """Apply a frozen role functional without identifiers or material truth."""
    return np.asarray(x) @ program["w"] + program["intercept"]
