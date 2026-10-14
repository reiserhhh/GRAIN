#!/usr/bin/env python3
"""Finite input-only actions and observation-preserving stresses for M05."""

from __future__ import annotations

import hashlib

import numpy as np

from .query_moment_operator import cell_residual

DIRECTIONS = (
    "F01",
    "F02",
    "F03",
    "F04",
    "within_neuroticism",
    "within_confirmation_seeking",
    "within_random",
)
STRESSES = ("cell_reverse", "cell_reflect")


def action_specs(epsilons=(0.02, 0.10)):
    """Enumerate a baseline, signed finite actions, and two nonlocal stresses."""
    epsilons = tuple(float(e) for e in epsilons)
    if (
        not epsilons
        or len(set(epsilons)) != len(epsilons)
        or any(not np.isfinite(e) or not 0 < e < 1 for e in epsilons)
    ):
        raise ValueError("Distinct finite relative amplitudes in (0,1) required")
    result = [{"action": "baseline", "kind": "baseline", "epsilon": 0.0, "sign": 0}]
    for name in DIRECTIONS:
        for e in epsilons:
            for sign in (-1, 1):
                result.append(
                    {
                        "action": f"{name}_{e:.6g}_{sign:+d}",
                        "kind": "finite",
                        "direction": name,
                        "epsilon": e,
                        "sign": sign,
                    }
                )
    result.extend(
        {"action": n, "kind": "stress", "epsilon": 0.0, "sign": 0} for n in STRESSES
    )
    return result


def cell_reverse(g, h):
    """Reverse state rows inside exact-g cells while retaining physical positions."""
    g, h = np.asarray(g), np.asarray(h)
    cell_residual(g, h)
    _, cells = np.unique(g, axis=0, return_inverse=True)
    result = h.copy()
    for cell in np.unique(cells):
        positions = np.flatnonzero(cells == cell)
        result[positions] = h[positions[::-1]]
    return result


def body_action(g, h, directions, spec):
    """Return the actual dtype-quantized BODY used by both explicit and native paths."""
    h = np.asarray(h)
    if h.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Floating BODY states required")
    if spec["kind"] == "baseline":
        result = h.copy()
    elif spec["kind"] == "finite":
        d = np.asarray(directions[spec["direction"]], float)
        if d.shape != h.shape or not np.isfinite(d).all():
            raise ValueError("Aligned finite direction required")
        result = h.astype(float) + spec["sign"] * spec["epsilon"] * d
    elif spec["action"] == "cell_reverse":
        result = cell_reverse(g, h)
    elif spec["action"] == "cell_reflect":
        result = h.astype(float) - 2 * cell_residual(g, h)
    else:
        raise ValueError("Unknown frozen action")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite intervention")
    return result.astype(h.dtype, copy=True)


def patch_body(state, positions, body):
    """Copy a full state and replace only ordered BODY rows, never final query row."""
    state, positions, body = np.asarray(state), np.asarray(positions), np.asarray(body)
    if state.ndim != 2 or positions.ndim != 1 or len(positions) < 2:
        raise ValueError("Full state and at least two ordered BODY rows required")
    if positions.dtype.kind not in "iu" or np.any(np.diff(positions) <= 0):
        raise ValueError("Distinct ordered integer positions required")
    if positions[0] < 0 or positions[-1] >= len(state) - 1:
        raise ValueError("BODY must not include final query position")
    if body.shape != (len(positions), state.shape[1]) or body.dtype != state.dtype:
        raise ValueError("Exact BODY shape and dtype required")
    if not np.isfinite(body).all() or not np.isfinite(state).all():
        raise ValueError("Finite full state and BODY required")
    result = state.copy()
    result[positions] = body
    return result


def array_hash(array):
    """Bind a numeric intervention to dtype, shape and contiguous bytes."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype.str}:{array.shape}:".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def odd_even(baseline, plus, minus):
    """Separate signed response from even curvature, without dividing by amplitude."""
    return (np.asarray(plus) - np.asarray(minus)) / 2, (
        np.asarray(plus) + np.asarray(minus)
    ) / 2 - np.asarray(baseline)


def quadratic_effect(g, h, d, model, epsilon):
    """Exact within-potential odd and even increments before dtype quantization."""
    r = cell_residual(g, h) @ model["query_basis"]
    rd = cell_residual(g, d) @ model["query_basis"]
    a, n = model["within_matrix"], len(h)
    return (
        2 * epsilon * np.einsum("ti,ij,tj->", r, a, rd) / n,
        epsilon**2 * np.einsum("ti,ij,tj->", rd, a, rd) / n,
    )
