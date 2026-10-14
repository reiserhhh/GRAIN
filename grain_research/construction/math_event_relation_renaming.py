#!/usr/bin/env python3
"""Finite occasion-code orbits with actor/event equality and order preserved."""

from __future__ import annotations

import copy
import itertools
import math
import re
from collections import Counter

import numpy as np

from grain_research.evidence_history import active_records

GROUP = (0, 1, 2, 3)
REGIMES = ("balanced", "alias_focus", "distinct_focus", "other_focus")
CONTRASTS = (
    ("alias_minus_balanced", "alias_focus", "balanced"),
    ("distinct_minus_alias", "distinct_focus", "alias_focus"),
    ("other_minus_balanced", "other_focus", "balanced"),
)
CODE = re.compile(r"\bK[1-4]\b")


def renamed(code, element):
    """Apply the declared cyclic action; reject undeclared codes or elements."""
    if element not in GROUP or CODE.fullmatch(code) is None:
        raise ValueError("Explicit K1..K4 and a declared C4 element required")
    return "K" + str(1 + (int(code[1]) - 1 + element) % 4)


def relation_signature(history):
    """Canonicalize occasion labels, retaining every other record field and order."""
    labels, result = {}, []
    for record in history:
        code = record["event_id"]
        if CODE.fullmatch(code) is None:
            raise ValueError("Unknown occasion code")
        labels.setdefault(code, len(labels))
        result.append({**record, "event_id": labels[code]})
    return result


def transform(body, history, element):
    """Consistently rename text and metadata, never replace record/actor identity."""
    if element not in GROUP or not history:
        raise ValueError("Nonempty declared history required")
    unknown = re.findall(r"\bK\d+\b", body)
    if not unknown or any(CODE.fullmatch(k) is None for k in unknown):
        raise ValueError("Text must contain only known occasion codes")
    if Counter(unknown) != Counter(r["event_id"] for r in history):
        raise ValueError("This finite roster requires one text occasion per record")
    changed = copy.deepcopy(history)
    for record in changed:
        record["event_id"] = renamed(record["event_id"], element)
    text = CODE.sub(lambda m: renamed(m.group(), element), body)
    if relation_signature(changed) != relation_signature(history):
        raise ValueError("Event equality/ownership/order changed")
    return text, changed


def actual_atom_weights(history, basis):
    """Count own records with their actual occasion code, not a shared K1 atom."""
    if basis == "event":
        records = [r for r in active_records(history)["records"] if r["actor"] == "A"]
    elif basis == "mention":
        records = [
            r for r in history if r["actor"] == "A" and r["kind"] in {"new", "alias"}
        ]
    else:
        raise ValueError("Declared event/mention basis required")
    if not records:
        raise ValueError("Absent evidence is not a zero score")
    counts = Counter((r["family"], r["polarity"], r["event_id"]) for r in records)
    keys = sorted(counts)
    return keys, [counts[k] / len(records) for k in keys], len(records)


def orbit_decomposition(values):
    """Orthogonally split a full C4 orbit; invariance is a construction identity."""
    x = np.asarray(values, float)
    if x.shape != (4,) or not np.isfinite(x).all():
        raise ValueError("Four finite orbit observations required")
    mean = math.fsum(x) / 4
    variance = math.fsum((v - mean) ** 2 for v in x) / 4
    energy = math.fsum(v * v for v in x) / 4
    return {
        "orbit_mean": mean,
        "code_variance": variance,
        "code_sd": math.sqrt(variance),
        "orbit_energy": energy,
        "invariant_energy": mean * mean,
        "identity_error": abs(energy - mean * mean - variance),
        "positive_codes": sum(v > 0 for v in x),
        "negative_codes": sum(v < 0 for v in x),
        "min": float(min(x)),
        "max": float(max(x)),
    }


def pair_design(targets):
    """Enumerate all matched regime and order pairs before reading any outputs."""
    pairs = []
    for content, family, polarity, order in itertools.product(
        targets, (0, 1), (0, 1), (0, 1)
    ):
        for contrast, hi, lo in CONTRASTS:
            pairs.append(
                {
                    "pair": len(pairs),
                    "content": content,
                    "family": family,
                    "polarity": polarity,
                    "order": order,
                    "contrast": contrast,
                    "hi_key": [content, family, polarity, order, hi],
                    "lo_key": [content, family, polarity, order, lo],
                    "orientation": 2 * polarity - 1,
                }
            )
    for content, family, polarity, regime in itertools.product(
        targets, (0, 1), (0, 1), REGIMES
    ):
        pairs.append(
            {
                "pair": len(pairs),
                "content": content,
                "family": family,
                "polarity": polarity,
                "order": -1,
                "contrast": "order_" + regime,
                "hi_key": [content, family, polarity, 1, regime],
                "lo_key": [content, family, polarity, 0, regime],
                "orientation": None,
            }
        )
    return pairs


def collision(summary_a, summary_b, native_a, native_b, values, tolerance):
    """Certify equality of a specified probability summary and its score floor."""
    a, b, n, m, v = (
        np.asarray(x, float) for x in (summary_a, summary_b, native_a, native_b, values)
    )
    if a.ndim != 1 or not (a.shape == b.shape == n.shape == m.shape == v.shape):
        raise ValueError("Aligned complete probability summaries required")
    if not all(np.isfinite(x).all() for x in (a, b, n, m, v)):
        raise ValueError("Finite collision inputs required")
    error = float(np.max(np.abs(a - b)))
    equal = error <= tolerance
    change = float((n - m) @ v)
    tv = float(np.abs(n - m).sum() / 2)
    return {
        "summary_max_difference": error,
        "same_summary": equal,
        "native_score_change": change,
        "native_tv": tv,
        "same_summary_changed_observation": equal and tv > tolerance,
        "paired_score_RMS_floor": abs(change) / 2 if equal else None,
        "paired_probability_L2_floor": float(np.linalg.norm(n - m)) / 2
        if equal
        else None,
    }
