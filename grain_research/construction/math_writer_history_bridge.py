#!/usr/bin/env python3
"""Distinguish current persona policy from an observed article's response action."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping

import numpy as np


def rendered_token_ids(rendered):
    """Accept old token lists and newer chat-template input dictionaries explicitly."""
    values = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(not isinstance(i, (int, np.integer)) for i in values)
    ):
        raise ValueError("Exactly one nonempty flat integer token sequence required")
    return list(map(int, values))


def choice_prompt(opportunity, order):
    """Render the same two actions in both label orders, without a trait query."""
    if order not in (0, 1):
        raise ValueError("Exactly two registered answer orders are supported")
    options = [opportunity["positive"], opportunity["negative"]]
    if order:
        options.reverse()
    return (
        "You now face a new choice. Choose what you would do next, rather than "
        "describing your traits or judging another person. "
        + opportunity["context"]
        + "\nA: "
        + options[0]
        + "\nB: "
        + options[1]
        + "\nReply with only A or B."
    )


def bridge_requests(pair, systems, articles, opportunities):
    """Bind nine real policy/history corners; never fabricate text/vector corners."""
    if set(systems) != {-1, 0, 1} or set(articles) != {-1, 0, 1}:
        raise ValueError("All three registered policy and history states required")
    if articles[0] != "":
        raise ValueError("Empty history is a registered control, not a new article")
    result = []
    for opportunity, order, policy, history in itertools.product(
        opportunities, (0, 1), (-1, 0, 1), (-1, 0, 1)
    ):
        unit = f"{pair['pair_id']}|{opportunity['id']}|o{order}|p{policy}|h{history}"
        result.append(
            {
                "request_id": unit,
                "pair_id": pair["pair_id"],
                "axis": pair["axis"],
                "situation": pair["situation"],
                "style": pair["style"],
                "seed": pair["seed"],
                "role": pair["role"],
                "policy": policy,
                "history": history,
                "opportunity": opportunity["id"],
                "domain": opportunity["domain"],
                "condition": opportunity["condition"],
                "order": order,
                "messages": [
                    {"role": "system", "content": systems[policy]},
                    {"role": "user", "content": pair["old_user"]},
                    {"role": "assistant", "content": articles[history]},
                    {"role": "user", "content": choice_prompt(opportunity, order)},
                ],
            }
        )
    return result


def choice_account(ab_logits, log_normalizer, order):
    """Retain grammar-conditioned choice AND unrestricted option availability."""
    ab = np.asarray(ab_logits, float)
    if ab.shape != (2,) or not np.isfinite(ab).all():
        raise ValueError("Two finite native token logits required")
    if order not in (0, 1) or not np.isfinite(log_normalizer):
        raise ValueError(
            "Finite full-vocabulary normalizer and registered order required"
        )
    shifted = np.exp(ab - ab.max())
    positive = order
    q = float(shifted[positive] / shifted.sum())
    log_mass = float(ab.max() + np.log(shifted.sum()) - log_normalizer)
    if log_mass > 2e-5:
        raise ValueError("Options have more than total vocabulary mass")
    return {
        "q_positive": q,
        "positive_log_odds": float(ab[positive] - ab[1 - positive]),
        "unrestricted_option_mass": float(np.exp(log_mass)),
    }


def sample_protocol_choice(q_positive, request_id, seed):
    """Draw one constrained response per registered event; do not multiply trials."""
    if not np.isfinite(q_positive) or not 0 <= q_positive <= 1:
        raise ValueError("A finite binary protocol law is required")
    key = hashlib.sha256(f"{seed}|{request_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    return int(rng.random() < q_positive)


def nine_corner_account(values):
    """Separate policy, actual article and their signed interaction at neutral origin."""
    q = np.asarray(values, float)
    if q.shape != (3, 3) or not np.isfinite(q).all():
        raise ValueError("Finite policy(-1,0,1) by history(-1,0,1) table required")
    reference = float(q[1, 1])
    policy = q[:, 1] - reference
    history = q[1, :] - reference
    interaction = q - reference - policy[:, None] - history[None, :]
    reconstructed = reference + policy[:, None] + history[None, :] + interaction
    return {
        "reference": reference,
        "policy_effect": float(policy[2] - policy[0]),
        "history_effect": float(history[2] - history[0]),
        "interaction_effect": float(interaction[2, 2] - interaction[0, 0]),
        "natural_effect": float(q[2, 2] - q[0, 0]),
        "history_under_high": float(q[2, 2] - q[2, 0]),
        "history_under_low": float(q[0, 2] - q[0, 0]),
        "joint_policy_history": float((q[2, 2] - q[2, 0] - q[0, 2] + q[0, 0]) / 4),
        "reconstruction_error": float(np.max(abs(q - reconstructed))),
    }


def response_witness(source_features, source_responses):
    """Freeze observed-action cross-moments; no labels, inverse or reader-answer fit."""
    x, y = np.asarray(source_features, float), np.asarray(source_responses, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Aligned source vectors and conditional choices required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite source observations required")
    origin, base = x.mean(0), y.mean(0)
    coefficients = (y - base).T @ (x - origin) / len(x)
    return origin, coefficients, np.linalg.norm(coefficients, axis=1)


def cold_response_reading(features, origin, coefficients, norms):
    """Apply an observed-choice witness to vectors only, not to persona metadata."""
    x = np.asarray(features, float)
    origin, c, n = map(np.asarray, (origin, coefficients, norms))
    if x.ndim != 2 or origin.shape != (x.shape[1],) or c.shape != (len(n), x.shape[1]):
        raise ValueError("Cold vector support differs from source")
    if not all(np.isfinite(v).all() for v in (x, origin, c, n)):
        raise ValueError("Finite cold program required")
    result = np.full((len(x), len(n)), np.nan)
    active = n > 0
    result[:, active] = (x - origin) @ c[active].T / n[active]
    return result
