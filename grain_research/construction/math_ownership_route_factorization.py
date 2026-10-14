#!/usr/bin/env python3
"""Disjoint-region C2 means and route-specific evidence observation accounts."""

from __future__ import annotations

import numpy as np

ARMS = (
    "body_even",
    "suffix_even",
    "both_even",
    "body_even_suffix_remove_both",
    "random_body",
    "random_body_suffix_remove_both",
)


def assignment(native, flipped, split, arm, low_suffix, g, seed):
    """Return exact cut positions/payload without aligning different x/y lengths."""
    a, b = np.asarray(native), np.asarray(flipped)
    s = np.asarray(low_suffix)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or a.dtype != np.float32
        or b.dtype != np.float32
        or not 0 < split < len(a)
        or s.shape != a[split:].shape
        or g not in (-1, 1)
        or arm not in (*ARMS, "noop")
        or not all(np.isfinite(v).all() for v in (a, b, s))
    ):
        raise ValueError("Matched finite float32 cuts and declared arm required")
    if arm == "noop":
        return np.arange(len(a)), a.copy()
    result = a.copy()
    pre = arm != "suffix_even"
    query = arm in (
        "suffix_even",
        "both_even",
        "body_even_suffix_remove_both",
        "random_body_suffix_remove_both",
    )
    if pre:
        mean = (a[:split].astype(float) + b[:split].astype(float)) / 2
        if arm.startswith("random_body"):
            direction = np.random.default_rng(seed).normal(size=mean.shape)
            direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
            norm = np.linalg.norm(
                a[:split].astype(float) - mean, axis=-1, keepdims=True
            )
            result[:split] = (a[:split].astype(float) - g * norm * direction).astype(
                np.float32
            )
        else:
            result[:split] = mean.astype(np.float32)
    if query:
        result[split:] = (
            ((a[split:].astype(float) + b[split:].astype(float)) / 2).astype(np.float32)
            if arm in ("suffix_even", "both_even")
            else s
        )
    positions = (
        np.arange(len(a))
        if pre and query
        else np.arange(split)
        if pre
        else np.arange(split, len(a))
    )
    return positions, result[positions]


def route_interaction(native, body_even, suffix_even, both_even):
    """Keep the signed nonadditive consequence of two disjoint cut operations."""
    arrays = [np.asarray(v, float) for v in (native, body_even, suffix_even, both_even)]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("Four matched response arrays required")
    return arrays[0] - arrays[1] - arrays[2] + arrays[3]


def toy_routes(kind):
    """Exhibit body, suffix and redundant ownership routes with equal native output."""
    g = np.array([-1, 1], float)
    readers = {
        "body": lambda b, s: b,
        "suffix": lambda b, s: s,
        "redundant": lambda b, s: b + s - b * b * s,
    }
    if kind not in readers:
        raise ValueError("Declared synthetic route required")
    read = readers[kind]
    return {
        "native": read(g, g),
        "body_even": read(np.zeros(2), g),
        "suffix_even": read(g, np.zeros(2)),
        "both_even": read(np.zeros(2), np.zeros(2)),
    }
