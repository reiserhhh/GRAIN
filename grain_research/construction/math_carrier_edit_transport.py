#!/usr/bin/env python3
"""Source-defined actor edits, with conditional and marginal finite actions."""

from __future__ import annotations

import numpy as np

REPLACEMENTS = (
    (
        "started conversations with several people at the gathering",
        "started discussions with several guests at the reception",
    ),
    (
        "stayed apart and spoke only when approached at the gathering",
        "stayed apart and replied only when questioned at the reception",
    ),
    (
        "worried about several possible problems before the appointment",
        "worried about several possible mistakes before the interview",
    ),
    (
        "remained calm about several possible problems before the appointment",
        "remained calm about several possible mistakes before the interview",
    ),
)


def new_body(text):
    """Apply a fixed new expression bank, without outcome-based editing."""
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError("Expected complete source expression")
        text = text.replace(old, new)
    return text


def edit_increment(cube, e, n, axis, marginal=False):
    """Return source actor-flip increment; optionally average the other role."""
    x = np.asarray(cube, dtype=np.float64)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete finite e/n memory support required")
    if e not in (-1, 1) or n not in (-1, 1) or axis not in (0, 1):
        raise ValueError("Invalid actor edit")
    i, j = (e + 1) // 2, (n + 1) // 2
    if axis == 0:
        return np.mean(x[1 - i] - x[i], axis=0) if marginal else x[1 - i, j] - x[i, j]
    return np.mean(x[:, 1 - j] - x[:, j], axis=0) if marginal else x[i, 1 - j] - x[i, j]


def apply_increment(base, increment, slots):
    """Add only to fixed carrier rows in float64, then cast once to native."""
    b, d = np.asarray(base), np.asarray(increment, dtype=np.float64)
    if b.shape != d.shape or b.ndim != 4 or not np.isfinite(d).all():
        raise ValueError("Aligned layer/head/time/channel increment required")
    if (
        not slots
        or len(set(slots)) != len(slots)
        or min(slots) < 0
        or max(slots) >= b.shape[2]
    ):
        raise ValueError("Invalid carrier support")
    result = b.copy()
    result[:, :, slots] = (b[:, :, slots].astype(np.float64) + d[:, :, slots]).astype(
        b.dtype
    )
    other = sorted(set(range(b.shape[2])) - set(slots))
    if not np.array_equal(result[:, :, other], b[:, :, other]):
        raise ValueError("Increment escaped its declared carrier")
    return result


def driver_coefficients(values):
    """Compute all actor and queried-person characters from complete readouts."""
    x = np.asarray(values, dtype=np.float64)
    if x.shape[:3] != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete e/n/person readout required")
    result = []
    for mask in range(8):
        total = np.zeros(x.shape[3:], dtype=np.float64)
        for i, e in enumerate((-1, 1)):
            for j, n in enumerate((-1, 1)):
                for k, p in enumerate((-1, 1)):
                    signs = (e, n, p)
                    weight = np.prod(
                        [s for bit, s in enumerate(signs) if mask & (1 << bit)]
                    )
                    total += weight * x[i, j, k]
        result.append(total / 8)
    return np.stack(result)
