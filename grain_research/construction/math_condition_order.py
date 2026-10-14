#!/usr/bin/env python3
"""Source-defined condition accounts and executable observation functions."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_author_contrast as A
from grain_research.construction import math_cold_joint_author as J

ARMS = ("marginal10", "joint49", "anchor_joint50", "conditional_joint99")
TARGETS = ("condition_choice", "focal_polarity", "conditional_polarity", "role_joint")


def matrices(g, q, joint, conditions):
    """Build four registered spans; c is supplied, never inferred or imputed."""
    g, q = J.finite_matrix(g, 4), J.finite_matrix(q, 6)
    joint = J.finite_matrix(joint, 49)
    c = np.asarray(conditions, float)
    if len(g) != len(q) or len(g) != len(joint):
        raise ValueError("same-event observation identities required")
    if c.shape != (len(g),) or not np.isin(c, [-1, 1]).all():
        raise ValueError("explicit aligned binary conditions required")
    return {
        "marginal10": np.column_stack((g, q)),
        "joint49": joint,
        "anchor_joint50": np.column_stack((joint, c)),
        "conditional_joint99": np.column_stack((joint, c, c[:, None] * joint)),
    }


def compile_function(features, targets, reference):
    """Least-squares calibrate design functions once in a frozen source law."""
    p = A.probability(reference)
    t = np.asarray(targets, float)
    if t.ndim != 2 or len(t) != len(p) or not np.isfinite(t).all():
        raise ValueError("source target columns required")
    projection = A.source_projection(features, p)
    keep = projection["singular"] > projection["tolerance"]
    coefficient = projection["right"].T @ (
        (projection["basis"].T @ (np.sqrt(p[:, None]) * t))
        / projection["singular"][keep, None]
    )
    beta = coefficient / projection["source_scale"][:, None]
    mean = p @ t
    offset = mean - projection["source_center"] @ beta
    predicted = np.asarray(features) @ beta + offset
    expected = A.project(t, p, projection) + mean
    error = float(np.max(abs(predicted - expected)))
    if error > 1e-10:
        raise ArithmeticError("source coefficient replay failed")
    return {
        "beta": beta.tolist(),
        "offset": offset.tolist(),
        "rank": projection["rank"],
        "source_replay_max_error": error,
        "calibration": "source design-function least squares; not neural or teacher fit",
    }


def order_account(features, reference, conditions):
    """Keep conditional-expectation E separate from multiplication M by c."""
    p = A.probability(reference)
    c = np.asarray(conditions, float)
    if c.shape != p.shape or not np.isin(c, [-1, 1]).all():
        raise ValueError("binary reference-aligned c required")
    if abs(p @ c) > 1e-12:
        raise ValueError("registered balanced binary reference required")
    projection = A.source_projection(features, p)
    root = np.sqrt(p)
    u = np.column_stack((root, projection["basis"]))
    P = u @ u.T
    E = np.outer(root, root) + np.outer(root * c, root * c)
    M = np.diag(c)
    pc = P @ (root * c) / root
    epc = E @ (root * pc) / root
    within = pc - epc
    visibility = A.norm2(pc, p)
    choice_to_within = (np.eye(len(p)) - E) @ P @ E
    within_to_choice = E @ P @ (np.eye(len(p)) - E)
    multiplication_leak = (np.eye(len(p)) - P) @ M @ u
    return {
        "rank_with_constant": int(u.shape[1]),
        "c_visibility": visibility,
        "choice_to_within_norm2": A.norm2(within, p),
        "predicted_choice_leak_norm2": visibility * (1 - visibility),
        "EPc_identity_error": float(np.max(abs(epc - visibility * c))),
        "E_commutator_Frobenius": float(np.linalg.norm(E @ P - P @ E)),
        "M_commutator_Frobenius": float(np.linalg.norm(M @ P - P @ M)),
        "choice_to_within_Frobenius": float(np.linalg.norm(choice_to_within)),
        "within_to_choice_Frobenius": float(np.linalg.norm(within_to_choice)),
        "M_span_escape_Frobenius": float(np.linalg.norm(multiplication_leak)),
        "rank_rule": "inherited fixed machine precision; no search",
    }


def cold_score(bundle, record):
    """Apply frozen coefficients to new numeric events; never read source truth."""
    if set(record) != {"event_id", "observer_id", "grain", "query", "condition"}:
        raise ValueError("only bound numeric evidence and condition are cold inputs")
    if not isinstance(record["event_id"], str) or not record["event_id"]:
        raise ValueError("explicit event address required")
    if record["observer_id"] != bundle["observation"]["observer_id"]:
        raise ValueError("source observer mismatch")
    joint, g, q = J.observed_features(
        bundle["observation"], [record["grain"]], [record["query"]]
    )
    c = record["condition"]
    if c is not None and (isinstance(c, bool) or c not in (-1, 1)):
        raise ValueError("unknown or explicit binary c required")
    base = {"marginal10": np.column_stack((g, q)), "joint49": joint}
    if c is not None:
        base = matrices(g, q, joint, [c])
    arms = {}
    for name in ARMS:
        if name not in base:
            arms[name] = {"status": "UNOBSERVED_CONDITION", "scores": None}
            continue
        compiled = bundle["functions"][name]
        value = base[name] @ np.asarray(compiled["beta"]) + compiled["offset"]
        if not np.isfinite(value).all():
            raise ArithmeticError("nonfinite cold function action")
        arms[name] = {"status": "OBSERVED_FUNCTION", "scores": value[0].tolist()}
    return {"event_id": record["event_id"], "arms": arms}
