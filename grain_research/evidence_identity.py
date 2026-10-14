#!/usr/bin/env python3
"""Typed linguistic evidence, context addresses, and finite mixture estimands.

Evidence coordinates are not traits or probabilities of actual behavior. Several
coordinates can be supported by one utterance. The simplex decoder is restricted
to a separately declared single-mode elicitation experiment, never open text.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import minimize

MODES = ("intention", "action_report", "preference", "ability", "experience", "request")
QUERIES = {
    "intention": "The respondent expresses their own intention or plan to do something.",
    "action_report": "The respondent reports an action that they themselves already performed.",
    "preference": "The respondent expresses their own preference or liking.",
    "ability": "The respondent claims that they themselves are capable of doing something.",
    "experience": "The respondent describes their own current feeling or emotional experience.",
    "request": "In this reply, the respondent is asking their conversation partner for help.",
    "repair": "In this reply, the respondent corrects a misunderstanding or a previous statement.",
    "disagree": "In this reply, the respondent disagrees with their conversation partner.",
    "accept": "In this reply, the respondent accepts their conversation partner's offer of help.",
    "content": "The conversation concerns asking for help or receiving assistance.",
}


def address(text, record_id, role):
    """Address an original turn without asserting an exact semantic subspan."""
    if not isinstance(text, str) or role not in {"respondent", "partner"}:
        raise ValueError("A raw text and explicit dialogue role are required")
    return {
        "record_id": str(record_id),
        "role": role,
        "start": 0,
        "end": len(text),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scope": "whole_turn_not_token_attribution",
    }


def requests(record_id, reply, partner="", *, include_context=True):
    """Create fixed queries from raw turns only, without experiment truth fields."""
    if not reply.strip():
        raise ValueError("Empty reply")
    premise = f"Respondent: {reply}"
    if include_context and partner:
        premise = f"Partner: {partner}\n{premise}"
    view = "context" if include_context else "reply_only"
    return [
        {
            "audit_id": f"{record_id}:{view}:{key}",
            "record_id": str(record_id),
            "view": view,
            "query": key,
            "sentence": premise,
            "item_text": hypothesis,
        }
        for key, hypothesis in QUERIES.items()
    ]


def evidence_record(record_id, reply, partner, probabilities):
    """Preserve independent NLI outputs and their role-bound textual support."""
    if set(probabilities) != set(QUERIES):
        raise ValueError("Complete fixed-query outputs required")
    values = {}
    for key, triple in probabilities.items():
        triple = np.asarray(triple, float)
        if (
            triple.shape != (3,)
            or not np.isfinite(triple).all()
            or np.any(triple < 0)
            or not np.isclose(triple.sum(), 1, atol=2e-6)
        ):
            raise ValueError("Expected entailment/neutral/contradiction probabilities")
        values[key] = {
            "entailment": float(triple[0]),
            "neutral": float(triple[1]),
            "contradiction": float(triple[2]),
            "margin": float(triple[0] - triple[2]),
            "supported_candidate": bool(triple[0] == triple.max()),
        }
    return {
        "record_id": str(record_id),
        "utterance": address(reply, record_id, "respondent"),
        "antecedent": address(partner, str(record_id) + ":previous", "partner")
        if partner
        else None,
        "antecedent_status": "supplied_previous_turn_not_resolved_target"
        if partner
        else "not_observed",
        "channels": values,
        "normalization": "independent_queries_not_exclusive_mode_probabilities",
        "real_world_action": "unobserved",
        "psychological_score": None,
        "denominator": "observed_utterance_only_no_unobserved_opportunities",
    }


def simplex_decode(observed, source_means):
    """Fit a finite mixture to source readings; apply only to the declared bank.

    This constrained least-squares reference estimates elicited statement-type
    composition, not an individual's intention-to-action conversion probability.
    """
    y, b = np.asarray(observed, float), np.asarray(source_means, float)
    if b.ndim != 2 or y.shape != (b.shape[1],) or not np.isfinite(b).all():
        raise ValueError(
            "Mode by query source matrix and matching observation required"
        )
    if not np.isfinite(y).all() or len(b) < 2:
        raise ValueError("Finite observation and at least two modes required")
    n = len(b)
    start = np.full(n, 1 / n)
    fit = minimize(
        lambda p: float(np.square(p @ b - y).sum()),
        start,
        jac=lambda p: 2 * b @ (p @ b - y),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={
            "type": "eq",
            "fun": lambda p: p.sum() - 1,
            "jac": lambda p: np.ones(n),
        },
        options={"maxiter": 200, "ftol": 1e-12},
    )
    if not fit.success or abs(fit.x.sum() - 1) > 1e-8:
        raise RuntimeError("Finite decoder failed: " + fit.message)
    return fit.x, float(np.linalg.norm(fit.x @ b - y))


def target_identifiability(source_means, target):
    """Test target membership in observed-query plus normalization row space."""
    b, q = np.asarray(source_means, float), np.asarray(target, float)
    if (
        b.ndim != 2
        or q.shape != (len(b),)
        or not np.isfinite(b).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("Finite mode by query matrix and one target per mode required")
    a = np.vstack([b.T, np.ones(len(b))])
    _, s, vh = np.linalg.svd(a, full_matrices=True)
    tolerance = max(a.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tolerance))
    projected = vh[:rank].T @ (vh[:rank] @ q)
    return {
        "rank": rank,
        "modes": len(b),
        "singular_values": s.tolist(),
        "target_null_component": float(np.linalg.norm(q - projected)),
        "target_identifiable_in_source_linear_model": bool(
            np.linalg.norm(q - projected) < 1e-9
        ),
        "null_basis": vh[rank:].tolist(),
    }


def opportunity_standardize(rates, weights):
    """Separate a fixed opportunity mix from an author's observed prompt mix."""
    rates, weights = np.asarray(rates, float), np.asarray(weights, float)
    if rates.shape[-1] != len(weights) or weights.ndim != 1:
        raise ValueError("One weight per observed condition required")
    if (
        not np.isfinite(rates).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("Finite rates and nonnegative weights required")
    if not np.isclose(weights.sum(), 1):
        raise ValueError("Opportunity weights must sum to one")
    return rates @ weights
