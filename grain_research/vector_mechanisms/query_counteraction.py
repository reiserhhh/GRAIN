#!/usr/bin/env python3
"""Local query contrast as carried input plus current-operator contrast."""

import numpy as np

from grain_research.vector_mechanisms.gated_update import (
    apply_coefficients,
    coefficients,
    rms_direction,
)


def exchange(x, d, weights):
    """Apply each query's actual local Jacobian to both queries' incoming tangents.

    Output axes are operator query, input query, product-rule term, factor,
    hidden coordinate. No residual identity is included in an MLP increment.
    """
    x, d = np.asarray(x, float), np.asarray(d, float)
    if (
        x.ndim != 2
        or d.ndim != 3
        or x.shape[0] != 2
        or d.shape[:2] != (2, 4)
        or x.shape[-1] != d.shape[-1]
        or not np.isfinite(x).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError("Two finite contexts and four aligned directions required")
    return np.stack(
        [
            np.stack(
                [
                    apply_coefficients(
                        coefficients(a, weights), rms_direction(a, b, weights), weights
                    )
                    for b in d
                ]
            )
            for a in x
        ]
    )


def decompose(values):
    """Exact symmetric accounting, keeping product-rule terms and signed vectors."""
    v = np.asarray(values, float)
    if v.ndim != 5 or v.shape[:4] != (2, 2, 2, 4) or not np.isfinite(v).all():
        raise ValueError("Expected operator/input/term/factor/vector exchange")
    carried = ((v[0, 0] - v[0, 1]) + (v[1, 0] - v[1, 1])) / 2
    operator = ((v[0, 0] - v[1, 0]) + (v[0, 1] - v[1, 1])) / 2
    return {"carried": carried, "operator": operator, "contrast": v[0, 0] - v[1, 1]}


def vector_check(actual, predicted):
    """Original native parity gate; zeros use absolute rather than relative error."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.shape != p.shape or a.ndim != 2 or not np.isfinite([a, p]).all():
        raise ValueError("Matched finite direction matrices required")
    e = np.linalg.norm(a - p, axis=-1)
    den = np.linalg.norm(a, axis=-1)
    rel = np.divide(e, den, out=np.zeros_like(e), where=den != 0)
    rel[(den == 0) & (e != 0)] = np.inf
    absolute = abs(a - p).max(axis=-1)
    return {
        "relative_max": float(rel.max()),
        "absolute_max": float(absolute.max()),
        "pass": bool(((rel <= 1e-4) | (absolute <= 1e-5)).all()),
    }
