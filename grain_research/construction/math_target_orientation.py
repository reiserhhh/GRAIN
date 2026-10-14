#!/usr/bin/env python3
"""Finite contrast-cone certificates and source-frozen numeric function actions."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

from grain_research.construction import math_cold_joint_author as J
from grain_research.construction import math_condition_order as C

SCOPES = ("all_contexts", "foreign_companion_anchor_self")
TOL = 5e-8


def contrast_rows(features, metadata):
    """Address all focal polarity pairs without collapsing context or targets."""
    b = np.asarray(features, float)
    if b.ndim != 2 or len(b) != len(metadata) or not np.isfinite(b).all():
        raise ValueError("aligned finite observations required")
    cells = {}
    for i, row in enumerate(metadata):
        key = (row["focal"], row["companion"], *row["coordinates"])
        if key in cells:
            raise ValueError("duplicate event design cell")
        cells[key] = i
    differences, addresses = [], []
    for focal, companion in itertools.permutations(J.TARGETS, 2):
        for c, actor, anchor in itertools.product((-1, 1), repeat=3):
            lo = (focal, companion, -1, c, actor, anchor)
            hi = (focal, companion, 1, c, actor, anchor)
            if lo not in cells or hi not in cells:
                raise ValueError("complete240 contrast support required")
            l, h = cells[lo], cells[hi]
            addresses.append(
                {
                    "focal": focal,
                    "companion": companion,
                    "c": c,
                    "actor": actor,
                    "anchor": anchor,
                    "low": l,
                    "high": h,
                }
            )
            differences.append(b[h] - b[l])
    return np.array(differences), addresses


def scope_indices(addresses, scope, target):
    """Choose declared design support, not a favorable outcome subset."""
    if scope not in SCOPES or target not in ("common", *J.TARGETS):
        raise ValueError("registered support and target required")
    return np.array(
        [
            i
            for i, r in enumerate(addresses)
            if (scope == SCOPES[0] or (r["actor"] == -1 and r["anchor"] == 1))
            and (target == "common" or r["focal"] == target)
        ],
        dtype=int,
    )


def orientation(differences):
    """Solve one min-L1 orientation or return a directly checkable Gordan witness."""
    d = np.asarray(differences, float)
    if d.ndim != 2 or min(d.shape) < 1 or not np.isfinite(d).all():
        raise ValueError("nonempty finite contrast matrix required")
    n, width = d.shape
    options = {"time_limit": 60, "maxiter": 5000, "primal_feasibility_tolerance": 1e-9}
    answer = linprog(
        np.ones(2 * width),
        A_ub=np.column_stack((-d, d)),
        b_ub=-np.ones(n),
        bounds=(0, None),
        method="highs",
        options=options,
    )
    if answer.success:
        w = answer.x[:width] - answer.x[width:]
        dual = -answer.ineqlin.marginals
        residuals = [
            max(0.0, 1 - float(np.min(d @ w))),
            max(0.0, -float(np.min(dual))),
            max(0.0, float(np.max(abs(d.T @ dual))) - 1),
            abs(float(sum(dual)) - float(sum(abs(w)))) / (1 + float(sum(abs(w)))),
        ]
        if max(residuals) > TOL:
            raise ArithmeticError("primal/dual orientation certificate failed")
        return {
            "status": "SOURCE_ORIENTATION_EXISTS",
            "w": w.tolist(),
            "dual": dual.tolist(),
            "L1": float(sum(abs(w))),
            "minimum_source_margin": float(np.min(d @ w)),
            "certificate_error": max(residuals),
            "solver_iterations": int(answer.nit),
        }
    if answer.status != 2:
        raise RuntimeError("orientation solver failed: " + answer.message)
    witness = linprog(
        np.zeros(n),
        A_eq=np.vstack((d.T, np.ones(n))),
        b_eq=np.r_[np.zeros(width), 1.0],
        bounds=(0, None),
        method="highs",
        options=options,
    )
    if not witness.success:
        raise ArithmeticError("infeasibility has no verified cancellation witness")
    lam = witness.x
    residual = max(
        float(np.max(abs(d.T @ lam))),
        abs(float(sum(lam)) - 1),
        max(0.0, -float(np.min(lam))),
    )
    if residual > TOL:
        raise ArithmeticError("cancellation witness failed")
    return {
        "status": "NO_COMMON_ORIENTATION_ON_DECLARED_SUPPORT",
        "w": None,
        "dual": lam.tolist(),
        "L1": None,
        "minimum_source_margin": None,
        "certificate_error": residual,
        "solver_iterations": int(answer.nit + witness.nit),
    }


def cold_score(bundle, record):
    """Apply all frozen functions; focal target and design x are not input fields."""
    if set(record) != {"event_id", "observer_id", "grain", "query", "condition"}:
        raise ValueError("only bound numeric observation and condition allowed")
    if record["observer_id"] != bundle["observation"]["observer_id"]:
        raise ValueError("observer identity mismatch")
    if not isinstance(record["event_id"], str) or not record["event_id"]:
        raise ValueError("event identity required")
    c = record["condition"]
    if c is not None and (isinstance(c, bool) or c not in (-1, 1)):
        raise ValueError("explicit binary or unknown condition required")
    joint, g, q = J.observed_features(
        bundle["observation"], [record["grain"]], [record["query"]]
    )
    arms = {"marginal10": np.column_stack((g, q)), "joint49": joint}
    if c is not None:
        arms = C.matrices(g, q, joint, [c])
    results = {}
    for key, function in bundle["functions"].items():
        if function["status"] not in (
            "SOURCE_ORIENTATION_EXISTS",
            "INHERITED_POOLED_CALIBRATION",
        ):
            results[key] = {"status": function["status"], "value": None}
        elif function["arm"] not in arms:
            results[key] = {"status": "UNOBSERVED_CONDITION", "value": None}
        else:
            value = float(arms[function["arm"]][0] @ function["w"] + function["offset"])
            if not np.isfinite(value):
                raise ArithmeticError("nonfinite compiled action")
            results[key] = {"status": "OBSERVED_FUNCTION", "value": value}
    return {"event_id": record["event_id"], "functions": results}


def author_laws(metadata, target, delta=0.5):
    """Construct two explicit evidence laws, not fictitious psychological labels."""
    if target not in J.TARGETS or not 0 < delta < 1:
        raise ValueError("declared target and interior difference required")
    h = np.array([r["x"] if r["focal"] == target else 0 for r in metadata], float)
    plus, minus = (1 + delta * h) / len(h), (1 - delta * h) / len(h)
    if abs(sum(plus) - 1) > 1e-12 or abs(sum(minus) - 1) > 1e-12:
        raise ValueError("balanced application support required")
    return plus, minus
