#!/usr/bin/env python3
"""Division-free model invariants for conditional construct-reading matrices.

These are algebraic response restrictions, not tests of probabilistic
independence and not covariance tetrads of a psychological factor model.
"""

from __future__ import annotations

import itertools

import numpy as np


def _ratio(left, right):
    """Normalize a polynomial discrepancy without dividing individual response cells."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    denominator = float(np.sum(left * left) + np.sum(right * right))
    return {
        "raw_polynomial_rms": float(np.sqrt(np.mean((left - right) ** 2))),
        "relative_polynomial_norm": float(
            np.linalg.norm(left - right) / np.sqrt(denominator)
        )
        if denominator
        else None,
        "left_right_squared_norm": denominator,
        "constraints": len(left),
    }


def restrictions(reference, condition):
    """Test scalar, row, column and two-sided multiplicative conditioning laws.

    The quartic restrictions are necessary for B=diag(r) A diag(t). When A has
    no zero entries they are also sufficient, since all minors of B/A vanish.
    At zero support they need not identify that model; retain that boundary.
    """
    a, b = np.asarray(reference, float), np.asarray(condition, float)
    if (
        a.shape != (6, 6)
        or b.shape != a.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Two complete finite six-by-six matrices required")
    pairs = tuple(itertools.combinations(range(6), 2))
    row_left, row_right, col_left, col_right = [], [], [], []
    left, right = [], []
    for i in range(6):
        for j, k in pairs:
            row_left.append(b[i, j] * a[i, k])
            row_right.append(b[i, k] * a[i, j])
            col_left.append(b[j, i] * a[k, i])
            col_right.append(b[k, i] * a[j, i])
    for i, k in pairs:
        for j, l in pairs:
            left.append(b[i, j] * b[k, l] * a[i, l] * a[k, j])
            right.append(b[i, l] * b[k, j] * a[i, j] * a[k, l])
    a2, b2 = float(np.sum(a * a)), float(np.sum(b * b))
    scalar = float(np.sum(a * b) / a2) if a2 else None
    residual = float(np.sum((b - scalar * a) ** 2)) if scalar is not None else None
    return {
        "reference_squared_norm": a2,
        "condition_squared_norm": b2,
        "reference_exact_zero_cells": int(np.sum(a == 0)),
        "condition_exact_zero_cells": int(np.sum(b == 0)),
        "scalar_projection": scalar,
        "scalar_residual_squared_share": residual / b2
        if b2 and residual is not None
        else None,
        "row_only": _ratio(row_left, row_right),
        "query_only": _ratio(col_left, col_right),
        "row_and_query": _ratio(left, right),
        "zero_support_caveat": bool(np.any(a == 0)),
    }
