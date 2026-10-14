#!/usr/bin/env python3
"""Finite carrier interventions and encoding/readout interaction accounting."""

from __future__ import annotations

import itertools

import numpy as np


def carrier_scopes(row):
    """Partition original clauses, other clauses, constant tail and descendants."""
    length = len(row["history"])
    names = ("extraversion", "neuroticism")
    own = [list(row["scopes"][name]) for name in names]
    for slots in own:
        if not slots or slots != list(range(min(slots), max(slots) + 1)):
            raise ValueError("Nonempty contiguous clause scopes required")
        if min(slots) < 0 or max(slots) >= length:
            raise ValueError("Clause outside native history")
    if set(own[0]) & set(own[1]):
        raise ValueError("Independent clause addresses must not overlap")
    tail = list(range(max(max(s) for s in own) + 1, length))
    if not tail:
        raise ValueError("A constant record ending is required")
    return {
        "e_local": own[0],
        "n_local": own[1],
        "e_other": own[1],
        "n_other": own[0],
        "e_closure": list(range(min(own[0]), length)),
        "n_closure": list(range(min(own[1]), length)),
        "tail": tail,
    }


def replace_slots(base, donor, slots):
    """Replace specified time rows only, preserving every other native value."""
    b, d = np.asarray(base), np.asarray(donor)
    if b.shape != d.shape or b.ndim != 4:
        raise ValueError("Aligned layer/head/time/channel memories required")
    if not np.isfinite(b).all() or not np.isfinite(d).all():
        raise ValueError("No nonfinite memory repair")
    if not slots or len(set(slots)) != len(slots):
        raise ValueError("Nonempty distinct memory addresses required")
    if min(slots) < 0 or max(slots) >= b.shape[2]:
        raise ValueError("Patch outside time dimension")
    result = b.copy()
    result[:, :, slots] = d[:, :, slots]
    other = sorted(set(range(b.shape[2])) - set(slots))
    if not np.array_equal(result[:, :, other], b[:, :, other]):
        raise ValueError("Patch escaped its carrier scope")
    return result


def additive_memory(base, e, n):
    """Combine two carrier increments in float64, not sequential overwrites."""
    b, x, y = (np.asarray(a, dtype=np.float64) for a in (base, e, n))
    if len({a.shape for a in (b, x, y)}) != 1 or not all(
        np.isfinite(a).all() for a in (b, x, y)
    ):
        raise ValueError("Aligned finite memories required")
    return x + y - b


def memory_characters(cube):
    """Retain all four e/n memory characters, including the mixed component."""
    x = np.asarray(cube, dtype=np.float64)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete finite e/n memory cube required")
    signs = tuple(itertools.product((-1, 1), repeat=2))
    return np.stack(
        [
            sum(
                np.prod([s[j] for j in range(2) if mask & (1 << j)]) * a
                for s, a in zip(signs, x.reshape(4, *x.shape[2:]), strict=True)
            )
            / 4
            for mask in range(4)
        ]
    )


def interaction_account(base, e, n, real_both, additive_both):
    """Separate carrier encoding from reader nonlinearity at a fixed reference."""
    b, x, y, r, a = (
        np.asarray(v, dtype=np.float64) for v in (base, e, n, real_both, additive_both)
    )
    if len({v.shape for v in (b, x, y, r, a)}) != 1 or not all(
        np.isfinite(v).all() for v in (b, x, y, r, a)
    ):
        raise ValueError("Aligned finite reader endpoints required")
    return {
        "total": r - x - y + b,
        "encoding": r - a,
        "reader": a - x - y + b,
    }


def json_native(value):
    """Convert finite numeric summaries to JSON-safe scalars without silent NaN."""
    if isinstance(value, dict):
        return {str(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        return json_native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite summary is not a JSON result")
    return value
