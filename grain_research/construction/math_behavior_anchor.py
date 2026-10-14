#!/usr/bin/env python3
"""Public behavior-key anchors and nonseparable complete-text comparisons."""

from itertools import product

import numpy as np
from scipy.optimize import linprog

from .math_balanced_evidence import slot_indicators


def fixed_family():
    """Choose one of each first-four event pairs; retain the last four events."""
    bits = np.array(list(product((0, 1), repeat=4)), dtype=int)
    removals = 2 * np.arange(4) + bits
    return bits, removals, slot_indicators(removals)


def public_directions(vectors, rows, targets):
    """Compile signed pole means from a public item key, without author labels."""
    h = np.asarray(vectors, float)
    if h.ndim != 2 or len(h) != len(rows) or not np.isfinite(h).all():
        raise ValueError("Complete finite item vectors required")
    direction = []
    for target in targets:
        high = [
            i
            for i, row in enumerate(rows)
            if row["target"] == target and row["pole"] == "high"
        ]
        low = [
            i
            for i, row in enumerate(rows)
            if row["target"] == target and row["pole"] == "low"
        ]
        if not high or not low:
            raise ValueError("Each public key needs both poles")
        d = h[high].mean(0) - h[low].mean(0)
        norm = np.linalg.norm(d)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Unobserved public behavior direction; no rescue")
        direction.append(d / norm)
    return np.asarray(direction)


def balanced_design(scores, conditions, *, target=3):
    """Create positive actual-text arms fixing event slots, rival means and count law."""
    g, c = np.asarray(scores, float), np.asarray(conditions)
    bits, removals, slots = fixed_family()
    if (
        g.shape != (16, 5)
        or not np.isfinite(g).all()
        or c.shape != (12,)
        or not np.isin(c, [0, 1]).all()
    ):
        raise ValueError(
            "Sixteen five-target observations and twelve binary conditions required"
        )
    if target not in range(5):
        raise ValueError("Unknown source target")
    kept = np.array([[j for j in range(12) if j not in s] for s in removals])
    counts = c[kept].sum(1)
    count_law = (counts[:, None] == np.arange(9)).astype(float)
    rivals = [q for q in range(5) if q != target]
    z = np.column_stack((slots.reshape(16, -1), g[:, rivals], count_law))
    scale = np.ptp(z, axis=0)
    scale[scale == 0] = 1
    zs = (z - z.mean(0)) / scale
    eq = np.vstack(
        (
            np.r_[np.ones(16), np.zeros(16)],
            np.r_[np.zeros(16), np.ones(16)],
            np.column_stack((zs.T, -zs.T)),
        )
    )
    rhs = np.r_[1.0, 1.0, np.zeros(z.shape[1])]
    objective = np.r_[-g[:, target], g[:, target]]
    result = linprog(
        objective,
        A_eq=eq,
        b_eq=rhs,
        bounds=(0, None),
        method="highs-ds",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success:
        raise RuntimeError(f"Fixed SOURCE design failed: {result.message}; no retry")
    p = result.x.reshape(2, 16)
    nu = p[0] - p[1]
    error = max(float(np.max(abs(eq @ result.x - rhs))), float(np.max(abs(nu @ z))))
    dual = np.asarray(result.eqlin.marginals)
    slack = objective - eq.T @ dual
    gap = float(objective @ result.x - rhs @ dual)
    if np.any(p < 0) or error > 1e-8 or abs(gap) > 1e-8 or np.min(slack) < -1e-8:
        raise ValueError("Invalid SOURCE law; no clipping or renormalization")
    return {
        "weights": p,
        "source_action": nu @ g,
        "balance_error": error,
        "count_distribution": p @ count_law,
        "dual_gap": gap,
        "dual": dual,
        "dual_min_slack": float(np.min(slack)),
        "constraints": z,
        "bits": bits,
    }


def walsh_accounts(values, weights):
    """Keep every signed order of the four-choice response under the actual law."""
    y, p = np.asarray(values, float), np.asarray(weights, float)
    bits, _, _ = fixed_family()
    if y.ndim == 1:
        y = y[:, None]
    if (
        y.ndim != 2
        or y.shape[0] != 16
        or p.shape != (2, 16)
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Complete finite response family required")
    characters = (-1.0) ** ((bits @ bits.T) % 2)
    coefficients = characters.T @ y / 16
    nu_character = (p[0] - p[1]) @ characters
    components = nu_character[:, None] * coefficients
    orders = bits.sum(1)
    accounts = np.array([components[orders == degree].sum(0) for degree in range(5)])
    return {
        "characters": characters,
        "coefficients": coefficients,
        "components": components,
        "by_order": accounts,
        "reconstruction": characters @ coefficients,
        "action": accounts.sum(0),
    }
