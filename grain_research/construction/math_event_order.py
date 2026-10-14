#!/usr/bin/env python3
"""Qualified order observations and context interactions, not fitted traits."""

from __future__ import annotations

import numpy as np

from suica_core.path_signature import piecewise_linear_signature


def validate_order(order, count=12):
    """Validate one permutation preserving the first four and final events."""
    raw = np.asarray(order)
    if raw.shape != (count,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("An integer complete event permutation is required")
    if sorted(raw.tolist()) != list(range(count)):
        raise ValueError("Event identity may not be added, removed or repeated")
    if count < 6 or raw[:4].tolist() != list(range(4)) or raw[-1] != count - 1:
        raise ValueError("Companion prefix4 and final event must be unchanged")
    return raw.astype(int)


def signature_levels(events):
    """Represent raw event vectors as cumulative increments using old Chen code."""
    z = np.asarray(events, float)
    if z.ndim != 2 or len(z) == 0 or not np.isfinite(z).all():
        raise ValueError("A nonempty finite event-vector sequence is required")
    d = z.shape[1]
    if not 1 <= d <= 15:
        raise ValueError("Use the frozen <=15-dimensional observation, not a grid")
    path = np.vstack((np.zeros(d), np.cumsum(z, axis=0)))
    flat = piecewise_linear_signature(path, depth=3)
    return (flat[:d], flat[d : d + d * d], flat[d + d * d :])


def second_level_direct(events):
    """Independently enumerate ordered pairs and within-event half tensors."""
    z = np.asarray(events, float)
    if z.ndim != 2 or len(z) == 0 or not np.isfinite(z).all():
        raise ValueError("A nonempty finite event-vector sequence is required")
    value = 0.5 * z.T @ z
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            value += np.outer(z[i], z[j])
    return value.ravel()


def ordered_interaction(selected, control, background, order):
    """Return all signature corners and the exact focal/background interaction."""
    s, c, b = (np.asarray(x, float) for x in (selected, control, background))
    p = validate_order(order, len(b))
    if s.ndim != 2 or c.shape != s.shape or b.shape[1:] != s.shape[1:]:
        raise ValueError("Matched focal arms and background dimensions required")
    corners = tuple(
        tuple(signature_levels(np.vstack((a, bg))) for a in (s, c)) for bg in (b, b[p])
    )
    mixed = tuple(
        (corners[0][0][k] - corners[0][1][k]) - (corners[1][0][k] - corners[1][1][k])
        for k in range(3)
    )
    b2_difference = second_level_direct(b) - second_level_direct(b[p])
    factorized = np.kron(s.sum(0) - c.sum(0), b2_difference)
    return {
        "corners": corners,
        "mixed": mixed,
        "background_second_difference": b2_difference,
        "factorized_third": factorized,
    }


def cold_account(value):
    """Accept only qualified vectors and their declared order, never outcomes."""
    if set(value) != {"author", "focal", "background", "order"}:
        raise ValueError("Only author, focal, background, order may enter")
    owner = value["author"]
    if not isinstance(owner, str) or not owner:
        raise ValueError("Declared author identity is required")
    vectors, seen = [], set()
    for name in ("focal", "background"):
        events = value[name]
        if not isinstance(events, list) or not events:
            raise ValueError("Both evidence blocks must be nonempty")
        block = []
        for event in events:
            if set(event) != {"event_id", "owner", "vector"}:
                raise ValueError("Only event identity, owner and vector may enter")
            eid = event["event_id"]
            if not isinstance(eid, str) or not eid or eid in seen:
                raise ValueError("Evidence identity must be distinct and nonempty")
            if event["owner"] != owner:
                raise ValueError("No silent mixed-author attribution")
            seen.add(eid)
            block.append(event["vector"])
        vectors.append(np.asarray(block, float))
    a, b = vectors
    p = validate_order(value["order"], len(b))
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Consistent event-vector dimensions required")
    before, after = (
        signature_levels(np.vstack((a, b))),
        signature_levels(np.vstack((a, b[p]))),
    )
    return {
        "author": owner,
        "mean_focal": a.mean(0).tolist(),
        "mean_background": b.mean(0).tolist(),
        "levels_original": [x.tolist() for x in before],
        "levels_reordered": [x.tolist() for x in after],
        "event_count": len(a) + len(b),
        "target_score": None,
        "not_physical_time_or_personality_inference": True,
    }
