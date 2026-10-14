#!/usr/bin/env python3
"""Finite report-edit consistency, distinct from temporal author dynamics."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_source_update_support import mask_memory
from grain_research.construction.math_target_carrier_closure import (
    carrier_scopes,
    memory_characters,
)

POLICIES = (
    "relation_other",
    "full_other",
    "relation_closure",
    "full_closure",
    "shared_other",
    "shared_closure",
)


def flip(state, axis):
    """Edit one attribution in a fixed report, not observe another event."""
    if tuple(state) not in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        raise ValueError("Two registered report-attribution signs required")
    if axis not in ("e", "n"):
        raise ValueError("Registered e/n report edit required")
    e, n = state
    return (-e, n) if axis == "e" else (e, -n)


def supports(row, policy):
    """Retain axis supports; share the mixed sector by union, never deletion."""
    if policy not in POLICIES:
        raise ValueError("No unregistered support search")
    scopes = carrier_scopes(row)
    kind = policy.split("_")[1]
    e, n = scopes["e_" + kind], scopes["n_" + kind]
    return e, n, sorted(set(e) | set(n))


def potential(coefficients, row, state, policy):
    """Compile a source state function with all marginal and mixed sectors."""
    flip(state, "e")
    if policy not in ("shared_other", "shared_closure"):
        raise ValueError("An axis-masked edge field is not assumed integrable")
    c = np.asarray(coefficients, dtype=np.float64)
    if c.ndim != 5 or c.shape[0] != 4 or not np.isfinite(c).all():
        raise ValueError("Four finite memory sectors required")
    se, sn, joint = supports(row, policy)
    e, n = state
    return (
        c[0]
        + e * mask_memory(c[1], se)
        + n * mask_memory(c[2], sn)
        + e * n * mask_memory(c[3], joint)
    )


def increment(coefficients, row, state, axis, policy):
    """Construct the registered edge, conditioned on the current attribution."""
    end = flip(state, axis)
    c = np.asarray(coefficients, dtype=np.float64)
    if c.ndim != 5 or c.shape[0] != 4 or not np.isfinite(c).all():
        raise ValueError("Four finite memory sectors required")
    if policy.startswith("shared_"):
        return potential(c, row, end, policy) - potential(c, row, state, policy)
    se, sn, _ = supports(row, policy)
    e, n = state
    raw = -2 * e * n * c[3]
    if policy.startswith("full_"):
        raw = raw - 2 * (e if axis == "e" else n) * c[1 if axis == "e" else 2]
    return mask_memory(raw, se if axis == "e" else sn)


def walk(coefficients, row, state, word, policy):
    """Accumulate counterfactual edits; repeated reports add no observations."""
    flip(state, "e")
    c = np.asarray(coefficients, dtype=np.float64)
    total = np.zeros_like(c[0])
    current = tuple(state)
    for axis in word:
        total += increment(c, row, current, axis, policy)
        current = flip(current, axis)
    return total, current


def predicted_loop(coefficients, row, state, policy):
    """Return the signed four-edit residue from shared-support compatibility."""
    flip(state, "e")
    c = np.asarray(coefficients, dtype=np.float64)
    se, sn, _ = supports(row, policy)
    if policy.startswith("shared_"):
        return np.zeros_like(c[3])
    e, n = state
    return 4 * e * n * (mask_memory(c[3], sn) - mask_memory(c[3], se))


def compile_cube(cube):
    """Compile observed source corners only, without endpoint responses or fit."""
    x = np.asarray(cube, dtype=np.float64)
    if x.ndim != 6 or x.shape[:2] != (2, 2):
        raise ValueError("Complete source attribution cube required")
    return memory_characters(x)
