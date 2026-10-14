#!/usr/bin/env python3
"""Finite content/owner key sectors and selective native intervention accounts."""

from __future__ import annotations

import itertools

import numpy as np


def key_characters(keys):
    """Decompose actual keys into constant, b, o and bo sectors, without fitting."""
    x = np.asarray(keys, dtype=np.float64)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete finite b/o keys required")
    s = np.array([-1.0, 1.0])
    return np.stack(
        (
            x.mean(axis=(0, 1)),
            np.einsum("b,bo...->...", s, x) / 4,
            np.einsum("o,bo...->...", s, x) / 4,
            np.einsum("b,o,bo...->...", s, s, x) / 4,
        )
    )


def reconstruct_key(sectors, b, o, role, joint, *, common_role=None):
    """Retain even keys while independently admitting role and bo components."""
    x = np.asarray(sectors, dtype=np.float64)
    if x.shape[0] != 4 or b not in (-1, 1) or o not in (-1, 1):
        raise ValueError("Four sectors and addressed signs required")
    if role not in (0, 1) or joint not in (0, 1) or not np.isfinite(x).all():
        raise ValueError("Finite binary component permissions required")
    r = x[2] if common_role is None else np.asarray(common_role, dtype=np.float64)
    if r.shape != x[2].shape or not np.isfinite(r).all():
        raise ValueError("Common role must share the exact registered support")
    return x[0] + b * x[1] + role * o * r + joint * b * o * x[3]


def intervention_parts(fields):
    """Separate blind, role, joint and their interaction on the complete grid."""
    x = np.asarray(fields, dtype=np.float64)
    if x.shape[0] != 4 or not np.isfinite(x).all():
        raise ValueError("Need 00,10,01,11 actual output fields")
    return np.stack((x[0], x[1] - x[0], x[2] - x[0], x[3] - x[1] - x[2] + x[0]))


def author_difference(beta, eta=0.5):
    """Integrate two same-marginal evidence laws, keeping both queried people."""
    x = np.asarray(beta, dtype=np.float64)
    if x.shape[0] != 8 or not np.isfinite(x).all() or not 0 < eta < 1:
        raise ValueError("Eight characters and a positive evidence law required")
    return np.stack([2 * eta * (x[3] + p * x[7]) for p in (-1, 1)])


def independent_characters(cube):
    """Explicitly average every role sign for an independent output verifier."""
    x = np.asarray(cube, dtype=np.float64)
    if x.shape[:3] != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete b/o/p support required")
    result = np.zeros((8, *x.shape[3:]))
    for bits in itertools.product(range(2), repeat=3):
        signs = [2 * j - 1 for j in bits]
        for mask in range(8):
            weight = np.prod([signs[j] for j in range(3) if mask & (1 << j)])
            result[mask] += weight * x[bits] / 8
    return result
