#!/usr/bin/env python3
"""Measured two-input contrasts and explicit allocation, without neural imports."""

from __future__ import annotations

import numpy as np

COMPONENTS = (
    "conditional_mean",
    "evidence_configuration",
    "query_state",
    "interaction",
)
ORDERS = ("evidence_then_query", "query_then_evidence", "symmetric")


def factorial_parts(cells):
    """Map columns f11/f01/f10/f00 to baseline/E/X/interaction, preserving F."""
    v = np.asarray(cells, float)
    if v.ndim != 2 or v.shape[1] != 4 or not len(v) or not np.isfinite(v).all():
        raise ValueError("Finite paired four-cell rows required")
    a, b, c, d = v.T
    parts = np.column_stack([d, c - d, b - d, a - c - b + d])
    if not np.allclose(parts.sum(1), a, atol=1e-12, rtol=0):
        raise ValueError("Four-cell reconstruction failed")
    return parts


def ordered_parts(parts, order):
    """Allocate interaction by two operation paths or their equal-weight average."""
    a = np.asarray(parts, float)
    if (
        a.ndim != 2
        or a.shape[1] != 4
        or not np.isfinite(a).all()
        or order not in ORDERS
    ):
        raise ValueError("Baseline/E/X/I columns and a declared order required")
    weight = {"evidence_then_query": 0.0, "query_then_evidence": 1.0, "symmetric": 0.5}[
        order
    ]
    return np.column_stack(
        [a[:, 0], a[:, 1] + weight * a[:, 3], a[:, 2] + (1 - weight) * a[:, 3]]
    )
