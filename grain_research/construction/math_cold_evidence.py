#!/usr/bin/env python3
"""Frozen body-only numerical attention without application query computation."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax


def rotate(raw, position, frequencies, scaling):
    """Place a frozen raw query using default half-paired RoPE, no neural calls."""
    q, freq = np.asarray(raw, float), np.asarray(frequencies, np.float32)
    if q.ndim != 2 or q.shape[1] != 2 * len(freq) or position < 0:
        raise ValueError("Compatible raw queries and nonnegative position required")
    angle = np.float32(position) * freq
    angle = np.concatenate((angle, angle))
    half = q.shape[1] // 2
    paired = np.concatenate((-q[:, half:], q[:, :half]), axis=1)
    return (q * np.cos(angle) + paired * np.sin(angle)) * scaling


def reading(keys, values, query, event_ids):
    """Return body-conditional total, event-uniform, pairing and query-free views."""
    k, v, q = (np.asarray(x, float) for x in (keys, values, query))
    ids = np.asarray(event_ids)
    if k.ndim != 3 or k.shape != v.shape or q.ndim != 2:
        raise ValueError("Aligned sequence/KV-head/dimension inputs required")
    if q.shape[1] != k.shape[2] or len(q) % k.shape[1]:
        raise ValueError("GQA head dimensions incompatible")
    if ids.shape != (len(k),) or set(ids.tolist()) != {0, 1, 2}:
        raise ValueError("Exactly three nonempty canonical event groups required")
    if not all(np.isfinite(a).all() for a in (k, v, q)):
        raise ValueError("Nonfinite evidence")
    ratio = len(q) // k.shape[1]
    repeated_v = np.repeat(v, ratio, axis=1)
    weights = np.stack([softmax(k[:, h // ratio] @ q[h]) for h in range(len(q))])
    total = np.einsum("hl,lhd->hd", weights, repeated_v)
    uniform = np.zeros_like(total)
    for event in range(3):
        ix = ids == event
        uniform += weights[:, ix].sum(1)[:, None] * repeated_v[ix].mean(0)
    return np.stack((total, uniform, total - uniform, repeated_v.mean(0)))


def all_queries(keys, values, event_ids, endpoint, bundle):
    """Compute every frozen target from one body carrier, without design truth."""
    return np.stack(
        [
            reading(
                keys,
                values,
                rotate(raw, endpoint + gap, bundle["frequencies"], bundle["scaling"]),
                event_ids,
            )
            for raw, gap in zip(bundle["raw_queries"], bundle["gaps"], strict=True)
        ]
    )


def polarized_action(high, low, query_high, query_low):
    """Split a native finite action symmetrically into carrier and query changes."""
    hh, hl = high(query_high), high(query_low)
    lh, ll = low(query_high), low(query_low)
    return (hh - lh + hl - ll) / 2, (hh - hl + lh - ll) / 2, hh - ll
