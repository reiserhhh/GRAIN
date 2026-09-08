#!/usr/bin/env python3
"""Minimum-change action constraints, not identification of a true gradient field."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.centered_adjoint import center_groups


def constrain_action(displacement, integrated_field, labels, endpoint_action):
    """Choose the Frobenius-nearest centered field with the given endpoint action."""
    d = center_groups(displacement, labels)
    g = center_groups(integrated_field, labels)
    if d.shape != g.shape or not np.isfinite(endpoint_action):
        raise ValueError("Finite matched state/field and endpoint action required")
    norm2 = float(np.sum(d**2))
    gap = float(endpoint_action - np.sum(d * g))
    if norm2 <= 1e-24:
        if abs(gap) > 1e-12:
            raise ValueError("Nonzero action cannot be imposed on a zero path")
        return g.copy(), {"gap_before": gap, "gap_after": gap, "correction_norm": 0.0}
    correction = gap * d / norm2
    result = g + correction
    return result, {
        "gap_before": gap,
        "gap_after": float(endpoint_action - np.sum(d * result)),
        "correction_norm": float(np.linalg.norm(correction)),
        "field_norm": float(np.linalg.norm(g)),
        "interpretation": "minimum-change representative; true field not identified",
    }
