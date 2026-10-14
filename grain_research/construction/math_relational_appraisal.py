#!/usr/bin/env python3
"""Finite relational appraisal designs and signed observation projections."""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np

from . import math_construct_routing as C


def documents(materials):
    """Cross behavioural binding, appraisal binding and full-record order."""
    rows = []
    for family, content, behaviour, appraisal, order in itertools.product(
        range(2), materials["targets"], range(2), ("none", "A", "B"), range(2)
    ):
        body = "This account reports actual actions by two anonymous people.\n"
        spans = {}
        actors = ("A", "B") if order == 0 else ("B", "A")
        for actor in actors:
            polarity = behaviour if actor == "A" else 1 - behaviour
            prop = materials["content_families"][content][family][polarity]
            body += f"Person {actor} "
            start = len(body)
            body += prop
            spans[actor] = [start, len(body)]
            body += ".\n"
            if appraisal != "none":
                word = "admirable" if actor == appraisal else "objectionable"
                body += f"A separate observer called Person {actor}'s action {word}.\n"
        body += "End of account."
        rows.append(
            {
                "doc_id": len(rows),
                "role": "source" if family == 0 else "application",
                "family": family,
                "content": content,
                "behaviour": behaviour,
                "appraisal": appraisal,
                "order": order,
                "first_actor": actors[0],
                "properties": spans,
                "body": body,
            }
        )
    if len(rows) != 144 or len({r["body"] for r in rows}) != 144:
        raise ValueError("Full unique144-account design required")
    return rows


def render(body, target, person, system, tokenizer):
    """Freeze one person-neutral query template; never truncate or repair results."""
    if person not in ("A", "B"):
        raise ValueError("A named target person is required")
    question = C.query(target).replace("the writer's", f"Person {person}'s")
    if "writer" in question:
        raise ValueError("Writer-only query is incompatible with the target person")
    user = "The following account describes two anonymous people.\n\n" + body
    user += "\n\nEvidence ends here.\n\nQuestion: " + question + "\nAnswer:"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    actual = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(actual, Mapping):
        actual = actual["input_ids"]
    if encoded["input_ids"] != actual or text.count(body) != 1:
        raise ValueError("Ambiguous actual prompt or tokenization")
    if len(actual) + 16 > 3072:
        raise ValueError("Keep full input; no truncation")
    return text, encoded, text.index(body)


def characters(dimension):
    """Return every character of a complete finite binary design."""
    if dimension not in (3, 4):
        raise ValueError("Three-bit control or four-bit main design required")
    bits = tuple(itertools.product(range(2), repeat=dimension))
    matrix = np.asarray(
        [
            [(-1) ** sum(a * b for a, b in zip(s, t, strict=True)) for t in bits]
            for s in bits
        ],
        float,
    )
    return bits, matrix


def decompose(values):
    """Retain every signed term, exact replay and simultaneous-name projection."""
    y = np.asarray(values, float)
    if y.ndim not in (3, 4) or y.shape != (2,) * y.ndim or not np.isfinite(y).all():
        raise ValueError("Complete finite observation cube required")
    bits, matrix = characters(y.ndim)
    # Remove the common level before projecting to avoid spurious null energy.
    mean = float(np.mean(y))
    beta = matrix @ (y.ravel() - mean) / y.size
    beta[0] = mean
    replay = (matrix.T @ beta).reshape(y.shape)
    odd = np.asarray([sum(bit) % 2 for bit in bits], bool)
    centered_energy = float(beta[1:] @ beta[1:])
    odd_energy = float(beta[odd] @ beta[odd])
    renamed = y[(slice(None, None, -1),) * y.ndim]
    return {
        "bits": bits,
        "coefficients": beta,
        "reconstruction_error": float(np.max(abs(y - replay))),
        "parseval_error": float(abs(np.mean(y * y) - beta @ beta)),
        "renaming_rms": float(np.sqrt(np.mean((y - renamed) ** 2))),
        "name_odd_squared_share": odd_energy / centered_energy
        if centered_energy
        else None,
        "centered_squared_energy": centered_energy,
    }


def bag_bound(scores):
    """Give the best constant squared loss on an actual identical-token-bag class."""
    y = np.asarray(scores, float)
    if y.ndim != 1 or y.size not in (4, 8) or not np.isfinite(y).all():
        raise ValueError("Four or eight actual equal-bag observations required")
    mean = float(np.mean(y))
    mse = float(np.mean((y - mean) ** 2))
    return {
        "rows": int(y.size),
        "oracle_group_mean": mean,
        "minimum_mse": mse,
        "minimum_rmse": float(np.sqrt(mse)),
        "range": float(np.ptp(y)),
    }
