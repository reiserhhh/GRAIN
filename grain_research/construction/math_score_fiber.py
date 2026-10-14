#!/usr/bin/env python3
"""Finite author-observation fibers, comparison bounds and relational readouts."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

SCHEMES = (
    "scores_only",
    "owner_count",
    "owner_response",
    "score_squares",
    "score_cross_products",
    "score_all_products",
)


def matrix(value):
    """Require finite two-dimensional data with a nonempty support."""
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError("Finite nonempty matrix required")
    return a


def probability(value, size):
    """Check a probability law without clipping or silently normalizing it."""
    p = np.asarray(value, dtype=float)
    if (
        p.shape != (size,)
        or not np.isfinite(p).all()
        or p.min() < -1e-12
        or abs(p.sum() - 1) > 1e-10
    ):
        raise ValueError("Nonnegative unit-mass law required")
    return p


def observation_design(scores, scheme):
    """Return SOURCE-defined event functions; products occur before averaging."""
    f = matrix(scores)
    if f.shape != (8, 6) or scheme not in SCHEMES:
        raise ValueError("All eight cells/six targets and a fixed scheme required")
    x = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
    columns = [np.ones(8), *f.T]
    if scheme == "owner_count":
        columns.append(x[:, 2])
    elif scheme == "owner_response":
        columns.append(x[:, 1] * x[:, 2])
    elif scheme == "score_squares":
        columns.extend(f.T**2)
    elif scheme in ("score_cross_products", "score_all_products"):
        pairs = (
            itertools.combinations(range(6), 2)
            if scheme == "score_cross_products"
            else itertools.combinations_with_replacement(range(6), 2)
        )
        columns.extend(f[:, i] * f[:, j] for i, j in pairs)
    return np.asarray(columns)


def row_chart(design):
    """Equilibrate known constraints and retain their exact numerical row span."""
    a = matrix(design)
    if np.max(abs(a[0] - 1)) > 1e-12:
        raise ValueError("First constraint must be unit mass")
    scales = np.linalg.norm(a, axis=1)
    scales[scales == 0] = 1
    u, s, vh = np.linalg.svd(a / scales[:, None], full_matrices=True)
    tolerance = 64 * np.finfo(float).eps * max(a.shape) * s[0]
    rank = int(np.sum(s > tolerance))
    transform = (u[:, :rank] / s[:rank]).T / scales[None, :]
    q, null = vh[:rank], vh[rank:].T
    if null.shape[1] == 1:
        h = null[:, 0]
        if h[np.argmax(abs(h))] < 0:
            null = -null
    return {
        "design": a,
        "basis": q,
        "transform": transform,
        "null": null,
        "rank": rank,
        "singular_values": s,
        "equilibration_error": float(np.max(abs(transform @ a - q))),
    }


def fiber_endpoints(profile, chart):
    """Attain the complete simplex fiber of a rank n or rank n-1 summary."""
    a, null = chart["design"], chart["null"]
    p = probability(profile, a.shape[1])
    if null.shape[1] == 0:
        return p.copy(), p.copy(), np.array([0.0, 0.0])
    if null.shape[1] != 1:
        raise ValueError("This finite protocol requires a zero/one-dimensional fiber")
    h = null[:, 0]
    active = abs(h) > 64 * np.finfo(float).eps
    positive, negative = (h > 0) & active, (h < 0) & active
    if not positive.any() or not negative.any():
        raise ArithmeticError("Unit-mass null direction must have both signs")
    limits = np.array(
        [np.max(-p[positive] / h[positive]), np.min(-p[negative] / h[negative])]
    )
    if limits[0] > 1e-10 or limits[1] < -1e-10:
        raise ArithmeticError("Observed profile must belong to its own fiber")
    lo, hi = p + limits[0] * h, p + limits[1] * h
    error = max(
        float(np.max(abs(a @ (lo - p)))),
        float(np.max(abs(a @ (hi - p)))),
        max(0.0, float(-min(lo.min(), hi.min()))),
    )
    if error > 1e-8:
        raise ArithmeticError("Unverified simplex fiber endpoints")
    return lo, hi, limits


def target_interval(values, endpoints):
    """Return sharp scalar/vector ranges and the endpoint that attains each."""
    f = np.asarray(values, dtype=float)
    p0, p1 = endpoints[:2]
    if f.ndim not in (1, 2) or f.shape[0] != len(p0) or not np.isfinite(f).all():
        raise ValueError("Aligned finite target observations required")
    v0, v1 = p0 @ f, p1 @ f
    return np.minimum(v0, v1), np.maximum(v0, v1), (v0 <= v1)


def comparison_interval(left, right):
    """Bound two independent authors, not two independent queries for one author."""
    return left[0] - right[1], left[1] - right[0]


def lp_interval(values, profile, chart):
    """Check sharp bounds by LP and save primal and original-unit dual evidence."""
    a, q, transform = chart["design"], chart["basis"], chart["transform"]
    p = probability(profile, a.shape[1])
    f = np.asarray(values, dtype=float)
    if f.shape != p.shape or not np.isfinite(f).all():
        raise ValueError("One finite scalar target per support cell required")
    b = q @ p
    answers = {}
    for side, sign in (("lower", 1), ("upper", -1)):
        fit = linprog(
            sign * f,
            A_eq=q,
            b_eq=b,
            bounds=(0, None),
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not fit.success:
            raise ArithmeticError(f"LP {side} failed:{fit.status}:{fit.message}")
        witness, dual = fit.x, sign * fit.eqlin.marginals
        original_dual = transform.T @ dual
        value = float(witness @ f)
        slack = f - q.T @ dual if side == "lower" else q.T @ dual - f
        error = max(
            float(np.max(abs(a @ (witness - p)))),
            max(0.0, float(-witness.min())),
            max(0.0, float(-slack.min())),
            abs(value - float(b @ dual)),
            abs(value - float((a @ p) @ original_dual)),
            float(np.max(abs(a.T @ original_dual - q.T @ dual))),
        )
        if error > 1e-8:
            raise ArithmeticError("Primal/dual certificate failed in original units")
        answers[side] = {
            "value": value,
            "p": witness,
            "dual": dual,
            "original_dual": original_dual,
            "error": error,
        }
    return answers


def character_basis():
    """Supply the fixed lexicographic x/y/g characters, not named personality axes."""
    bits = np.asarray(list(itertools.product((0, 1), repeat=3)))
    cells = 2 * bits - 1
    return np.prod(np.where(bits[None] == 1, cells[:, None], 1), axis=2) / np.sqrt(
        8
    ), bits.sum(1)


def cofactor_null(design):
    """Independently obtain a rank7-by8 null vector from alternating minors."""
    a = matrix(design)
    if a.shape != (7, 8):
        raise ValueError("Seven independent constraints on eight cells required")
    balanced = a / np.linalg.norm(a, axis=1)[:, None]
    h = np.asarray(
        [(-1.0) ** j * np.linalg.det(np.delete(balanced, j, axis=1)) for j in range(8)]
    )
    norm = np.linalg.norm(h)
    if not np.isfinite(norm) or norm == 0:
        raise ArithmeticError("Cofactors did not identify a null direction")
    return h / norm
