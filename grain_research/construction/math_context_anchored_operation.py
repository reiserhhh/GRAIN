#!/usr/bin/env python3
"""HM77: a source-fixed joint operation and a three-cause transport account."""

from __future__ import annotations

import numpy as np

from . import math_source_ownership_operator as S

METHODS = (
    "global_orthogonal",
    "global_oblique",
    "anchored_orthogonal",
    "anchored_oblique",
)


def anchor(model, body_rows):
    """Predict the query-even displacement from one body's pre-query state."""
    body = S.finite_rows(body_rows)
    if body.shape[1] != len(model["body_origin"]):
        raise ValueError("Body coordinate dimension changed")
    return ((body - model["body_origin"]) @ model["anchor_right"].T) @ model[
        "anchor_left"
    ].T


def fit(body_plus, body_minus, query_plus, query_minus):
    """Fit aligned SOURCE pairs only, keeping all numerical dimensions fixed."""
    bp, bm, qp, qm = map(
        S.finite_rows, (body_plus, body_minus, query_plus, query_minus)
    )
    if not (bp.shape == bm.shape == qp.shape == qm.shape):
        raise ValueError("Aligned source body/query partners required")
    body_models, body_info = S.fit(bp, bm)
    query_models, query_info = S.fit(qp, qm)
    if "oblique" not in body_models or "oblique" not in query_models:
        raise ValueError(
            "Source even/odd direct sums required; no regularization rescue"
        )
    be, bd = (bp + bm) / 2, (bp - bm) / 2
    qe = (qp + qm) / 2
    b0, q0 = be.mean(0), qe.mean(0)
    eb, _, _ = S.span(be - b0)
    db, _, _ = S.span(bd)
    if eb.shape[1] == 0:
        raise ValueError("No source context variation to anchor")
    c = np.column_stack((eb, db))
    u, singular, vt = np.linalg.svd(c, full_matrices=False)
    tolerance = max(c.shape) * np.finfo(float).eps * singular[0]
    if np.sum(singular > tolerance) != c.shape[1]:
        raise ValueError("Body anchor has no identified direct-sum extension")
    right = ((vt.T / singular) @ u.T)[: eb.shape[1]]
    coordinates = (be - b0) @ eb
    mapping, _, rank, _ = np.linalg.lstsq(coordinates, qe - q0, rcond=None)
    if rank != eb.shape[1]:
        raise ValueError("Source even coordinate rank changed")
    left = mapping.T
    models = {}
    for method, query_model in query_models.items():
        models[method] = query_model | {
            "body_origin": b0,
            "body_left": body_models["oblique"]["left"],
            "body_right": body_models["oblique"]["right"],
            "anchor_left": left,
            "anchor_right": right,
        }
    model = models["oblique"]
    info = {
        "source_pairs": len(bp),
        "body": body_info,
        "query": query_info,
        "anchor_rank": int(rank),
        "source_even_fit_max_error": float(np.max(abs(anchor(model, be) - (qe - q0)))),
        "source_odd_anchor_max_error": float(np.max(abs((bd @ right.T) @ left.T))),
        "anchor_body_projector_factor_max_error": float(
            np.max(abs(right @ model["body_left"]))
        ),
        "oblique_query_anchor_factor_max_error": float(
            np.max(abs(model["right"] @ left))
        ),
    }
    return models, info


def operate(model, body, query, *, conditioned):
    """Read ONE text's two states; never read a partner, design label or answer."""
    b, q = np.asarray(body, float), np.asarray(query, float)
    if b.ndim != 1 or q.ndim != 1 or b.shape != q.shape:
        raise ValueError("One aligned body/query pair required")
    if not np.isfinite(b).all() or not np.isfinite(q).all():
        raise ValueError("Finite input states required")
    if len(q) != len(model["origin"]):
        raise ValueError("Query coordinate dimension changed")
    center = model["origin"] + (anchor(model, b[None])[0] if conditioned else 0)
    new_q = q - 2 * S.project(model, (q - center)[None])[0]
    new_b = b - 2 * model["body_left"] @ (
        model["body_right"] @ (b - model["body_origin"])
    )
    return new_b, new_q


def accounts(model, bp, bm, qp, qm, *, conditioned):
    """Decompose partner error into context, missed action and anchor contamination."""
    bp, bm, qp, qm = map(S.finite_rows, (bp, bm, qp, qm))
    if not (bp.shape == bm.shape == qp.shape == qm.shape):
        raise ValueError("Aligned evaluation body/query partners required")
    qe, qd = (qp + qm) / 2 - model["origin"], (qp - qm) / 2
    be, bd = (bp + bm) / 2, (bp - bm) / 2
    predicted_even = anchor(model, be) if conditioned else np.zeros_like(qe)
    predicted_odd = (
        (bd @ model["anchor_right"].T) @ model["anchor_left"].T
        if conditioned
        else np.zeros_like(qd)
    )
    context = S.project(model, qe - predicted_even)
    missing = qd - S.project(model, qd)
    contamination = S.project(model, predicted_odd)
    sq = lambda value: np.sum(value * value, axis=1)
    cross = 2 * np.sum(missing * contamination, axis=1)
    return {
        "context_sq": sq(context),
        "missing_sq": sq(missing),
        "contamination_sq": sq(contamination),
        "odd_cross": cross,
        "operator_error_sq": 4 * (sq(context) + sq(missing + contamination)),
        "identity_error_sq": 4 * sq(qd),
        "anchor_even_prediction_error_sq": sq(qe - predicted_even),
        "anchor_actual_odd_sq": sq(predicted_odd),
    }


def summarize(rows):
    """Report mechanism accounts, with cross terms explicit for oblique operations."""
    total, identity = (
        np.mean(rows["operator_error_sq"]),
        np.mean(rows["identity_error_sq"]),
    )
    result = {
        "pairs": len(rows["operator_error_sq"]),
        "relative_error": float(np.sqrt(total / identity)) if identity else None,
        "operator_error_rms": float(np.sqrt(total)),
        "identity_error_rms": float(np.sqrt(identity)),
        "pairs_better_than_identity": int(
            np.sum(rows["operator_error_sq"] < rows["identity_error_sq"])
        ),
    }
    for key in ("context_sq", "missing_sq", "contamination_sq", "odd_cross"):
        result[key + "_mean"] = float(np.mean(rows[key]))
        result[key + "_fraction"] = (
            float(4 * np.mean(rows[key]) / total) if total else None
        )
    result["anchor_actual_odd_rms"] = float(
        np.sqrt(np.mean(rows["anchor_actual_odd_sq"]))
    )
    return result
