#!/usr/bin/env python3
"""Cross-query interventions in one shared endpoint coordinate, not token-row alignment."""

from __future__ import annotations

import numpy as np


def partition_routes(target, source):
    """Partition fixed source candidates by membership in the target candidate set."""
    keys = {(c["site"], c["head"]) for c in target}
    return {
        "shared": [c for c in source if (c["site"], c["head"]) in keys],
        "source_specific": [c for c in source if (c["site"], c["head"]) not in keys],
    }


def endpoint_delta_patch(target, source_origin, source_donor, routes, head_dim):
    """Add a source edit's final-slot write to the target's unedited final slot.

    Every query may have a different P length. Only the last site row, Q, is
    shared by position role. Layer and head coordinates remain model-identical.
    This intervention is a numerical transport hypothesis, not a semantic identity.
    """
    patches, seen = {}, set()
    if head_dim < 1:
        raise ValueError("Positive head dimension required")
    for c in routes:
        site, head = c["site"], int(c["head"])
        if (site, head) in seen:
            raise ValueError("Duplicate route")
        seen.add((site, head))
        a, b, t = (
            np.asarray(s["sites"][site]) for s in (source_origin, source_donor, target)
        )
        if any(
            v.ndim != 2 or not len(v) or not np.isfinite(v).all() for v in (a, b, t)
        ):
            raise ValueError("Finite site arrays required")
        if a.shape[1] != b.shape[1] or a.shape[1] != t.shape[1]:
            raise ValueError("Model coordinate mismatch")
        if head < -1 or (head + 1) * head_dim > t.shape[1]:
            raise ValueError("Head out of range")
        sl = (
            slice(None) if head == -1 else slice(head * head_dim, (head + 1) * head_dim)
        )
        if site not in patches:
            patches[site] = t.copy()
        patches[site][-1, sl] += b[-1, sl] - a[-1, sl]
    return patches


def linear_cka(x, y):
    """Compare centered event-response representations; undefined constants return NaN."""
    x, y = (np.asarray(v, float) for v in (x, y))
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Matched event matrices required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite event matrices required")
    x, y = x - x.mean(0), y - y.mean(0)
    k, l = x @ x.T, y @ y.T
    denominator = np.linalg.norm(k) * np.linalg.norm(l)
    return float(np.sum(k * l) / denominator) if denominator > 1e-20 else float("nan")
