#!/usr/bin/env python3
"""Source-only C2 operations and their independently measurable closure defects."""

from __future__ import annotations

import numpy as np

from .math_heterogeneous_evidence_cube import TARGETS, grid
from .math_passive_event_composition import action_line

METHODS = ("orthogonal", "oblique")


def finite_rows(values):
    """Require a nonempty finite matrix rather than accepting score metadata."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError("Nonempty finite vector rows required")
    return a


def span(rows):
    """Use the complete numerical span with a fixed machine-precision tolerance."""
    a = finite_rows(rows)
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    tol = max(a.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tol))
    return vt[:rank].T, s, tol


def fit(plus, minus):
    """Construct two involutions from source pairs, without held-out outcomes."""
    plus, minus = finite_rows(plus), finite_rows(minus)
    if plus.shape != minus.shape:
        raise ValueError("Comparable source partners required")
    even, odd = (plus + minus) / 2, (plus - minus) / 2
    origin = even.mean(0)
    qe, se, te = span(even - origin)
    qo, so, to = span(odd)
    if qo.shape[1] == 0:
        raise ValueError("No source ownership action observed")
    c = np.column_stack((qe, qo))
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    tol = max(c.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tol))
    result = {
        "orthogonal": {"origin": origin, "left": qo, "right": qo.T},
    }
    if rank == c.shape[1]:
        inverse = (vt.T / s) @ u.T
        result["oblique"] = {
            "origin": origin,
            "left": qo,
            "right": inverse[qe.shape[1] :],
        }
    diagnostics = {
        "source_pairs": len(plus),
        "dimension": plus.shape[1],
        "even_rank": qe.shape[1],
        "odd_rank": qo.shape[1],
        "joint_rank": rank,
        "joint_columns": c.shape[1],
        "oblique_constructed": "oblique" in result,
        "even_tolerance": float(te),
        "odd_tolerance": float(to),
        "joint_tolerance": float(tol),
        "even_min_kept": float(se[qe.shape[1] - 1]) if qe.shape[1] else None,
        "odd_min_kept": float(so[qo.shape[1] - 1]),
        "joint_min_singular": float(s[-1]),
        "joint_condition": float(s[0] / s[-1]) if s[-1] else None,
    }
    return result, diagnostics


def project(model, vectors):
    """Apply a source-frozen odd projector to arbitrary centered vector rows."""
    h = finite_rows(vectors)
    left, right = model["left"], model["right"]
    if h.shape[1] != len(model["origin"]):
        raise ValueError("Frozen coordinate dimension changed")
    return (h @ right.T) @ left.T


def operate(model, vector):
    """Compute the partner estimate from ONE vector, with no labels or partner."""
    h = np.asarray(vector, float)
    if h.ndim != 1 or not np.isfinite(h).all():
        raise ValueError("Exactly one finite application vector required")
    return h - 2 * project(model, (h - model["origin"])[None])[0]


def pair_accounts(model, plus, minus):
    """Measure transport separately from the exact involution's group law."""
    plus, minus = finite_rows(plus), finite_rows(minus)
    if plus.shape != minus.shape:
        raise ValueError("Comparable evaluation partners required")
    even = (plus + minus) / 2 - model["origin"]
    odd = (plus - minus) / 2
    leakage = project(model, even)
    missing = odd - project(model, odd)
    denominator = np.sum(odd * odd, axis=1)
    leakage_sq = np.sum(leakage * leakage, axis=1)
    missing_sq = np.sum(missing * missing, axis=1)
    residual_sq = 4 * (leakage_sq + missing_sq)
    return {
        "even_leakage_sq": leakage_sq,
        "odd_missing_sq": missing_sq,
        "identity_error_sq": 4 * denominator,
        "operator_error_sq": residual_sq,
        "even_leakage": leakage,
        "odd_missing": missing,
    }


def summarize(accounts):
    """Report finite vector-action ratios, not predictive personality accuracy."""
    identity = float(np.mean(accounts["identity_error_sq"]))
    operator = float(np.mean(accounts["operator_error_sq"]))
    return {
        "pairs": len(accounts["identity_error_sq"]),
        "identity_error_rms": float(np.sqrt(identity)),
        "operator_error_rms": float(np.sqrt(operator)),
        "relative_error": float(np.sqrt(operator / identity)) if identity else None,
        "even_leakage_error_rms": float(
            2 * np.sqrt(np.mean(accounts["even_leakage_sq"]))
        ),
        "odd_missing_error_rms": float(
            2 * np.sqrt(np.mean(accounts["odd_missing_sq"]))
        ),
        "pairs_better_than_identity": int(
            np.sum(accounts["operator_error_sq"] < accounts["identity_error_sq"])
        ),
        "even_fraction_of_error": float(
            4 * np.mean(accounts["even_leakage_sq"]) / operator
        )
        if operator
        else None,
    }


def fresh_documents(spec):
    """Make complete new cyclic-pair cells and exact sentence-order controls."""
    if set(spec["behaviours"]) != set(TARGETS):
        raise ValueError("All six fixed targets required")
    result = []
    for pair_id, focal in enumerate(TARGETS):
        companion = TARGETS[(pair_id + 1) % len(TARGETS)]
        for order in (0, 1):
            for corner, (x, y, g) in enumerate(grid(3)):
                actor, alternate = ("A", "B") if g == 1 else ("B", "A")
                lines = [
                    action_line("K1", "A", spec["behaviours"][focal][int(x > 0)]),
                    action_line("K2", actor, spec["behaviours"][companion][int(y > 0)]),
                    action_line("K3", alternate, spec["anchor"]),
                ]
                if order:
                    lines.reverse()
                result.append(
                    {
                        "doc_id": len(result),
                        "pair_id": pair_id,
                        "focal": focal,
                        "companion": companion,
                        "order": order,
                        "corner": corner,
                        "coordinates": [int(x), int(y), int(g)],
                        "body": "\n".join(lines),
                    }
                )
    return result
