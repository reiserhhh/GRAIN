#!/usr/bin/env python3
"""Finite direct-action channel capacity with checkable LP dual bounds."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

from grain_research.construction.math_cold_joint_author import TARGETS
from grain_research.construction.math_target_orientation import orientation

TOL = 2e-7


def spaces(marginal, joint, actors):
    """Retain raw observations and add only declared companion-owner products."""
    g = np.asarray(actors, float)
    if g.ndim != 1 or not np.isin(g, (-1, 1)).all():
        raise ValueError("binary declared companion actor required")
    result = {}
    for name, raw in (("marginal10", marginal), ("joint49", joint)):
        b = np.asarray(raw, float)
        if b.ndim != 2 or len(b) != len(g) or not np.isfinite(b).all():
            raise ValueError("aligned finite observations required")
        result[name] = b
        result["owner_" + name] = np.column_stack((b, g, g[:, None] * b))
    return result


def contrasts(features, metadata, target):
    """Address60 own and20 foreign contrasts, never treating notebook as query."""
    b = np.asarray(features, float)
    if target not in TARGETS or b.ndim != 2 or len(b) != len(metadata):
        raise ValueError("declared target and aligned observations required")
    cells = {}
    for i, r in enumerate(metadata):
        key = (r["focal"], r["companion"], *r["coordinates"])
        if key in cells:
            raise ValueError("duplicate design cell")
        cells[key] = i
    own, foreign, address = [], [], []
    for focal, companion in itertools.permutations(TARGETS, 2):
        if target not in (focal, companion):
            continue
        axis = 0 if focal == target else 1
        for fixed in itertools.product((-1, 1), repeat=3):
            lo, hi = list(fixed), list(fixed)
            lo.insert(axis, -1)
            hi.insert(axis, 1)
            l = cells[(focal, companion, *lo)]
            h = cells[(focal, companion, *hi)]
            qualifies = axis == 0 or lo[2] == 1
            difference = b[h] - b[l]
            (own if qualifies else foreign).append(difference)
            address.append(
                {
                    "kind": "own" if qualifies else "foreign",
                    "target": target,
                    "focal": focal,
                    "companion": companion,
                    "low": l,
                    "high": h,
                    "companion_actor": lo[2],
                    "notebook_actor": lo[3],
                    "query_person": "A",
                    "changed_event": "K1" if axis == 0 else "K2",
                }
            )
    if (len(own), len(foreign)) != (60, 20):
        raise ValueError("complete own60/foreign20 support required")
    return np.array(own), np.array(foreign), address


def capacity(own, foreign):
    """Minimize foreign action at fixed own orientation and certify the bound."""
    a, n = np.asarray(own, float), np.asarray(foreign, float)
    if (
        a.ndim != 2
        or n.ndim != 2
        or min(a.shape) < 1
        or min(n.shape) < 1
        or a.shape[1] != n.shape[1]
        or not np.isfinite(a).all()
        or not np.isfinite(n).all()
    ):
        raise ValueError("nonempty aligned finite contrast matrices required")
    na, nn, width = len(a), len(n), a.shape[1]
    matrix = np.vstack(
        (
            np.column_stack((-a, np.zeros(na))),
            np.column_stack((n, -np.ones(nn))),
            np.column_stack((-n, -np.ones(nn))),
        )
    )
    rhs = np.r_[-np.ones(na), np.zeros(2 * nn)]
    answer = linprog(
        np.r_[np.zeros(width), 1.0],
        A_ub=matrix,
        b_ub=rhs,
        bounds=[(None, None)] * width + [(0, None)],
        method="highs",
        options={
            "time_limit": 60,
            "maxiter": 5000,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    if not answer.success:
        if answer.status != 2:
            raise RuntimeError("capacity solver failed:" + answer.message)
        witness = orientation(a)
        if witness["w"] is not None:
            raise ArithmeticError("finite capacity infeasible despite own orientation")
        return {"status": "NO_OWN_ORIENTATION", "w": None, "witness": witness}
    w, bound = answer.x[:-1], float(answer.x[-1])
    dual = -answer.ineqlin.marginals
    lam, positive, negative = dual[:na], dual[na : na + nn], dual[na + nn :]
    residual = max(
        max(0.0, 1 - float(np.min(a @ w))),
        max(0.0, float(np.max(abs(n @ w))) - bound),
        max(0.0, -float(np.min(dual))),
        max(0.0, float(sum(positive + negative)) - 1),
        float(np.max(abs(a.T @ lam - n.T @ (positive - negative)))),
        abs(float(sum(lam)) - bound),
    )
    if residual > TOL * (1 + abs(bound)):
        raise ArithmeticError("source capacity primal/dual certificate failed")
    return {
        "status": "CAPACITY_BOUND_CERTIFIED",
        "w": w.tolist(),
        "foreign_action_bound": bound,
        "minimum_own_action": float(np.min(a @ w)),
        "lambda": lam.tolist(),
        "mu_positive": positive.tolist(),
        "mu_negative": negative.tolist(),
        "coefficient_L1": float(sum(abs(w))),
        "certificate_error": residual,
        "iterations": int(answer.nit),
    }


def evaluate(functions, observed):
    """Apply every target function and preserve complete contextual observations."""
    out = {}
    for key, fn in functions.items():
        b = np.asarray(observed[fn["space"]], float)
        if fn["w"] is None:
            out[key] = None
        else:
            out[key] = (b @ fn["w"]).tolist()
    return {"direct_channels": out, "context_retained": observed}
