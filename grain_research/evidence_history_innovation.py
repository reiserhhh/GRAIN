#!/usr/bin/env python3
"""Square-root form of the frozen history model's conditional innovation.

The original batch estimates remain unchanged. This companion avoids subtracting
nearly equal covariance entries when two reports almost duplicate an event.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr, solve_triangular


def source_coordinates(model, records):
    """Express prior, block, event and report variation as independent coordinates."""
    c = np.asarray(model["components"], float)
    if c.shape != (5,) or np.any(c <= 0) or not np.isfinite(c).all():
        raise ValueError("Five positive frozen variance components required")
    z = np.asarray([r["condition"] for r in records], float)
    x = np.c_[np.ones(len(z)), z]
    parts = [x * np.sqrt(c[:2])]
    for field, variance in zip(["block_id", "event_id", "record_id"], c[2:]):
        ids = [
            r.get("origin_id", r[field]) if field == "record_id" else r[field]
            for r in records
        ]
        unique = list(dict.fromkeys(ids))
        parts.append(
            np.asarray([[i == k for k in unique] for i in ids], float)
            * np.sqrt(variance)
        )
    h = np.column_stack(parts)
    target = np.zeros((2, h.shape[1]))
    target[:, :2] = np.diag(np.sqrt(c[:2]))
    return x, h, target


def square_root_append(model, old_values, old_records, new_values, new_record):
    """Project a new observation outside the old row span before taking its norm.

    Requires independent report IDs in the active old ledger. A new exact alias
    is supported, but contradictory values on that alias raise an error.
    """
    rows = old_records + [new_record]
    x, h, target = source_coordinates(model, rows)
    y = np.atleast_2d(np.asarray(old_values, float))
    v = np.asarray(new_values, float).reshape(-1)
    mean = np.asarray(model["mean"], float)
    if (
        y.shape[1] != len(old_records)
        or len(y) != len(v)
        or not np.isfinite(y).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("Finite old/new rows for the same author batch required")
    q, r = qr(h[:-1].T, mode="economic")
    if r.size and np.min(np.abs(np.diag(r))) < 1e-12:
        raise ValueError("Collapse exact old aliases before square-root updating")
    alpha = solve_triangular(r.T, (y - x[:-1] @ mean).T, lower=True)
    old_target = mean + (target @ q @ alpha).T
    residual_target = target - (target @ q) @ q.T
    residual = h[-1] - q @ (q.T @ h[-1])
    variance = float(residual @ residual)
    innovation = v - (x[-1] @ mean + h[-1] @ q @ alpha)
    cross = residual_target @ residual
    if variance < 1e-20:
        if np.max(np.abs(innovation)) > 1e-9:
            raise ValueError("Conflicting exact alias")
        gain = np.zeros(2)
    else:
        gain = cross / variance
    updated_root = residual_target - np.outer(gain, residual)
    return {
        "mean": old_target + innovation[:, None] * gain,
        "covariance": updated_root @ updated_root.T,
        "gain": gain,
        "innovation": innovation,
        "innovation_variance": variance,
    }
