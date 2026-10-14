#!/usr/bin/env python3
"""Finite actor-even/odd behavior fields and untuned expression correspondence."""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

PERMUTATIONS = np.array(list(itertools.permutations(range(6))), dtype=int)


def material(spec, variant, wording, family, condition, polarity, owner):
    """Keep both persons at the same opportunity and supply explicit action owner."""
    if any(s not in (-1, 1) for s in (condition, polarity, owner)):
        raise ValueError("Declared binary signs required")
    phrase = spec["conditions"][wording][(condition + 1) // 2]
    actor, other = ("A", "B") if owner == 1 else ("B", "A")
    action = spec["behaviours"][variant][family][(polarity + 1) // 2]
    return {
        "condition_text": "\n".join(
            f"On occasion K1, Person {p} {phrase}." for p in ("A", "B")
        ),
        "action_text": f"On occasion K1, Person {actor} {action}.\nPerson {other} recorded the date.",
        "author": "A",
        "actor": actor,
        "same_actor_opportunity": True,
    }


def actor_fields(values):
    """Resolve behavior effects into assignment-even/odd parts, retaining signs."""
    x = np.asarray(values, float)
    if x.ndim < 4 or x.shape[-4:-1] != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Finite [...,condition,polarity,owner,coordinate] required")
    gain = (x[..., 1, :, :] - x[..., 0, :, :]) / 2
    own, other = gain[..., 1, :], gain[..., 0, :]
    return {
        "own": own,
        "other": other,
        "even": (own + other) / 2,
        "odd": (own - other) / 2,
    }


def direction_match(source, target, min_norm=1e-12, tie_atol=1e-12):
    """Compare predetermined six-family identities with all720 label matchings."""
    s, t = np.asarray(source, float), np.asarray(target, float)
    if s.shape != t.shape or s.ndim != 3 or s.shape[:2] != (6, 2):
        raise ValueError("Same six-family/two-condition coordinate space required")
    if not np.isfinite(s).all() or not np.isfinite(t).all():
        raise ValueError("Finite effects required")
    sn, tn = np.linalg.norm(s, axis=2), np.linalg.norm(t, axis=2)
    unavailable = (sn <= min_norm) | (tn <= min_norm)
    if unavailable.any():
        return {
            "status": "UNAVAILABLE_DIRECTION",
            "unavailable_family_condition": np.argwhere(unavailable).tolist(),
            "matched_mean_cosine": None,
            "permutation_values": None,
        }
    cross = np.einsum("fcq,gcq->cfg", s / sn[..., None], t / tn[..., None])
    matched = np.stack([cross[c].diagonal() for c in range(2)], axis=1)
    values = cross[:, np.arange(6)[None, :], PERMUTATIONS].mean(axis=(0, 2))
    actual = float(matched.mean())
    return {
        "status": "OBSERVED_DIRECTION",
        "matched_mean_cosine": actual,
        "matched_family_condition": matched.tolist(),
        "permutation_mean": float(values.mean()),
        "permutation_max": float(values.max()),
        "matching_rank_with_ties": int(np.sum(values >= actual - tie_atol)),
        "finite_permutation_tail": float(np.mean(values >= actual - tie_atol)),
        "matched_minus_permutation_mean": actual - float(values.mean()),
        "permutation_values": values.tolist(),
        "source_norms": sn.tolist(),
        "target_norms": tn.tolist(),
    }


def lexical_fields(token_pairs):
    """Construct role-blind count differences in a fixed design-token vocabulary."""
    if len(token_pairs) != 3 or any(len(v) != 6 for v in token_pairs):
        raise ValueError("Three variants and six families required")
    vocabulary = sorted({t for v in token_pairs for f in v for b in f for t in b})
    index = {token: i for i, token in enumerate(vocabulary)}
    gains = np.zeros((3, 6, len(vocabulary)))
    for v, f, b in itertools.product(range(3), range(6), range(2)):
        for token, count in Counter(token_pairs[v][f][b]).items():
            gains[v, f, index[token]] += (2 * b - 1) * count / 2
    return np.repeat(gains[:, :, None], 2, axis=2), vocabulary


def author_contrasts(field, amplitude=0.4):
    """Integrate paired one-family contrasts under equal family/condition mass."""
    x = np.asarray(field, float)
    if x.ndim != 3 or x.shape[:2] != (6, 2) or not 0 < amplitude <= 1:
        raise ValueError("Six families/two conditions and legal amplitude required")
    return 2 * amplitude * x.mean(1) / 6
