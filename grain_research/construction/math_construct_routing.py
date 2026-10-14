#!/usr/bin/env python3
"""Finite multi-query response geometry on counterfactual evidence cubes.

Characters refer to controlled design indices; they are not assumed to act
linearly on natural-language hidden states. Scores remain reader outputs.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np

from grain_research.vector_mechanisms.direct_traits import BIG5, SYSTEM, question

BITS = tuple(itertools.product(range(2), repeat=3))
AXES = ("polarity", "other", "imagined")
TERMS = tuple(
    "*".join(a for a, bit in zip(AXES, b, strict=True) if bit) or "mean" for b in BITS
)


def query(target):
    """Use existing Big Five questions and one fixed narrow sixth query."""
    if target in BIG5:
        return question(target)
    if target == "anthropomorphism":
        return (
            "Estimate the writer's tendency to attribute human-like feelings, wishes, "
            "or intentions to nonhuman objects or software. Use 0 for very low, 9 "
            "for very high, and the digits in between for intermediate tendencies. "
            "Return only one digit from 0 to 9."
        )
    raise ValueError("Unknown frozen target")


def material(property_text, other, imagined, wording):
    """Keep the behavioural clause identical while changing explicit evidence context."""
    if any(v not in (0, 1) for v in (other, imagined, wording)):
        raise ValueError("Binary account fields required")
    owner = "my colleague" if other else "me"
    if wording == 0:
        status = (
            "only imagined; it did not happen"
            if imagined
            else "an actual event; it happened"
        )
        prefix = f"I am the writer of this account. The person described is {owner}. This was {status}. The person "
    else:
        status = (
            "an imagined scenario, not an actual event"
            if imagined
            else "an event that actually occurred"
        )
        prefix = f"In this account I describe {status}. The subject is {owner}, and that person "
    return prefix + property_text + ". End of account.", (
        len(prefix),
        len(prefix) + len(property_text),
    )


def documents(spec):
    """Declare both example families, forms and every cube cell before collection."""
    if list(spec["content_families"]) != spec["targets"] or len(spec["targets"]) != 6:
        raise ValueError("Six ordered fixed targets required")
    rows = []
    for family in (0, 1):
        for content in spec["targets"]:
            for wording in (0, 1):
                for polarity, other, imagined in BITS:
                    pair = spec["content_families"][content][family]
                    if len(pair) != 2 or any(
                        not isinstance(x, str) or not x for x in pair
                    ):
                        raise ValueError("Two complete behavioural clauses required")
                    prop = pair[polarity]
                    body, interval = material(prop, other, imagined, wording)
                    rows.append(
                        {
                            "doc_id": len(rows),
                            "role": "source" if family == 0 else "application",
                            "family": family,
                            "content": content,
                            "wording": wording,
                            "polarity": polarity,
                            "other": other,
                            "imagined": imagined,
                            "property": prop,
                            "property_interval": list(interval),
                            "body": body,
                        }
                    )
    return rows


def render(body, target, tokenizer):
    """Address the real chat tokens, preserving complete text and query suffix."""
    user = "The following is an account written by one anonymous writer.\n\n" + body
    user += "\n\nEvidence ends here.\n\nQuestion: " + query(target) + "\nAnswer:"
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    actual = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(actual, Mapping):
        actual = actual["input_ids"]
    if encoded["input_ids"] != actual or text.count(body) != 1:
        raise ValueError("Ambiguous or inconsistent actual template")
    if len(actual) + 16 > 3072:
        raise ValueError("Do not silently truncate controlled input")
    return text, encoded, text.index(body)


def token_positions(offsets, start, stop):
    """Use all tokens intersecting a nonempty character interval."""
    positions = [
        i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < stop
    ]
    if not positions:
        raise ValueError("Empty token interval")
    return positions


def characters():
    """Return the balanced eight characters of the experimental cube."""
    return np.asarray(
        [
            [(-1) ** sum(a * b for a, b in zip(s, t, strict=True)) for t in BITS]
            for s in BITS
        ],
        float,
    )


def split_cube(values):
    """Retain every term, checking reconstruction and Parseval independently."""
    x = np.asarray(values, dtype=float)
    if x.shape[:3] != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete finite polarity/ownership/actuality cube required")
    flat = x.reshape((8,) + x.shape[3:])
    beta = np.tensordot(characters() / 8, flat, axes=(1, 0))
    replay = np.tensordot(characters().T, beta, axes=(1, 0)).reshape(x.shape)
    return beta, {
        "reconstruction_max_error": float(np.max(abs(x - replay))),
        "parseval_error": float(abs(np.sum(x * x) / 8 - np.sum(beta * beta))),
    }


def spectrum(matrix):
    """Describe a finite response matrix, never an author-factor covariance matrix."""
    a = np.asarray(matrix, float)
    if a.shape != (6, 6) or not np.isfinite(a).all():
        raise ValueError("Six by six finite response matrix required")
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    energy = float(s @ s)
    total = float(s.sum())
    p = s / total if total else s
    effective_rank = (
        float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))) if total else 0.0
    )
    leading = float(s[0] ** 2 / energy) if energy else None
    valence = np.array([1, -1, 1, 1, 1, 0], float)
    valence /= np.linalg.norm(valence)
    v_energy = float(np.sum(np.outer(valence, valence @ a) ** 2))
    return {
        "singular_values": s.tolist(),
        "squared_energy": energy,
        "leading_squared_share": leading,
        "effective_rank": effective_rank,
        "evaluative_content_axis_squared_share": v_energy / energy if energy else None,
        "diagonal_mean": float(np.trace(a) / 6),
        "off_diagonal_rms": float(np.sqrt(np.sum((a - np.diag(np.diag(a))) ** 2) / 30)),
        "left_leading": u[:, 0].tolist(),
        "right_leading": vt[0].tolist(),
    }


def norm_matched(delta, seed):
    """Make a deterministic random intervention with the donor-delta norm, not a fit."""
    a = np.asarray(delta, float)
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError("Finite nonempty donor delta required")
    random = np.random.default_rng(seed).normal(size=a.shape)
    return random * (np.linalg.norm(a) / np.linalg.norm(random))


def patch_pairs(rows):
    """Declare application-only surface0 bidirectional role and actuality interventions."""
    lookup = {(r["doc_id"], r["query"]): r for r in rows}
    docs = {r["doc_id"]: r for r in rows}
    pairs = []
    for d in docs.values():
        if d["role"] != "application" or d["wording"] != 0:
            continue
        if d["imagined"] == 0:
            axes = ["other"]
        else:
            axes = []
        if d["other"] == 0:
            axes += ["imagined"]
        for axis in axes:
            donor = next(
                r
                for r in docs.values()
                if all(
                    r[k] == (1 - d[k] if k == axis else d[k])
                    for k in (
                        "family",
                        "content",
                        "wording",
                        "polarity",
                        "other",
                        "imagined",
                    )
                )
            )
            a, b = (
                lookup[d["doc_id"], d["content"]],
                lookup[donor["doc_id"], d["content"]],
            )
            if (
                a["property_tokens"] != b["property_tokens"]
                or a["property"] != b["property"]
            ):
                raise ValueError(
                    "Only identical property bytes/tokens can be exchanged"
                )
            pairs.append(
                {
                    "pair_id": len(pairs),
                    "axis": axis,
                    "receiver_unit": a["unit"],
                    "donor_unit": b["unit"],
                }
            )
    if len(pairs) != 48:
        raise ValueError("All48 predeclared application comparisons required")
    return pairs
