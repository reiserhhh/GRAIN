#!/usr/bin/env python3
"""Explicit property spans and source-only owner-displacement interventions."""

from __future__ import annotations

import re

import numpy as np

SENTENCE = re.compile(
    r"(?P<actor>I|[Mm]y colleague) "
    r"(?P<verb>felt|judged AI systems generally) "
    r"(?P<property>[^.]+?) while (?P<other>I|[Mm]y colleague) "
    r"(?P<neutral>[^.]+)\."
)


def selected_masks():
    """Return two crossed four-corner designs at both orders, with their union."""
    affect = {2 + a + 4 * u + 16 * s for a in (0, 1) for u in (0, 1) for s in (0, 1)}
    trust = {2 * t + 8 * v + 16 * s for t in (0, 1) for v in (0, 1) for s in (0, 1)}
    return sorted(affect | trust)


def owner_pairs():
    """List the eight other-to-self donor interventions per material family."""
    return [
        {
            "axis": axis,
            "polarity": polarity,
            "order": order,
            "receiver_mask": base + bit,
            "donor_mask": base,
        }
        for axis, bit, offset, pbit in (("A", 4, 2, 1), ("T", 8, 0, 2))
        for polarity in (0, 1)
        for order in (0, 1)
        for base in (offset + pbit * polarity + 16 * order,)
    ]


def raw_spans(text):
    """Locate declared actors/properties/neutral actions without an LLM parser."""
    result = {}
    for match in SENTENCE.finditer(text):
        axis = "A" if match.group("verb") == "felt" else "T"
        if f"{axis}_property" in result:
            raise ValueError("repeated property relation is outside this design")
        for name in ("actor", "property", "neutral"):
            lo, hi = match.span(name)
            result[f"{axis}_{name}"] = {"lo": lo, "hi": hi, "raw": text[lo:hi]}
    if set(result) != {
        f"{a}_{n}" for a in ("A", "T") for n in ("actor", "property", "neutral")
    }:
        raise ValueError("both explicit relation spans are required")
    return result


def token_span(rendered, body_offset, span, encoding, question_start):
    """Bind a raw span to intersecting tokens, admitting only whitespace overlap."""
    lo, hi = body_offset + span["lo"], body_offset + span["hi"]
    if rendered[lo:hi] != span["raw"]:
        raise ValueError("raw property address changed")
    positions = []
    for i, (start, end) in enumerate(encoding["offset_mapping"]):
        if end <= lo or start >= hi:
            continue
        if (start < lo and rendered[start:lo].strip()) or (
            end > hi and rendered[hi:end].strip()
        ):
            raise ValueError("token crosses a semantic span boundary")
        if i >= question_start or end <= start:
            raise ValueError("span enters query or has no raw character address")
        positions.append(i)
    if not positions:
        raise ValueError("empty token span")
    return {
        **span,
        "positions": positions,
        "token_ids": [encoding["input_ids"][i] for i in positions],
    }


def shared_basis(affect_deltas, trust_deltas):
    """Span two mean owner displacements using source data only, without rank tuning."""
    a, t = np.asarray(affect_deltas, float), np.asarray(trust_deltas, float)
    if (
        a.ndim != 2
        or t.ndim != 2
        or a.shape[1] != t.shape[1]
        or not np.isfinite(a).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError("finite source owner-displacement rows required")
    directions = np.column_stack((a.mean(axis=0), t.mean(axis=0)))
    u, singular, _ = np.linalg.svd(directions, full_matrices=False)
    tolerance = np.finfo(float).eps * max(directions.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    return u[:, :rank], directions, singular


def random_basis(dimension, rank, seed):
    """Create a deterministic, label-independent orthonormal comparison frame."""
    if not 0 <= rank <= dimension:
        raise ValueError("invalid random-frame rank")
    return np.linalg.qr(
        np.random.default_rng(seed).normal(size=(dimension, rank)), mode="reduced"
    )[0]


def intervention_delta(current, donor, arm, basis, random_frame):
    """Split actual donor displacement; random control matches projected token norms."""
    x, y, u, r = map(np.asarray, (current, donor, basis, random_frame))
    if (
        x.ndim != 2
        or x.shape != y.shape
        or u.shape[0] != x.shape[1]
        or r.shape != u.shape
    ):
        raise ValueError("matched property matrices and frames required")
    if not all(np.isfinite(v).all() for v in (x, y, u, r)):
        raise ValueError("finite interventions required")
    for frame in (u, r):
        if not np.allclose(frame.T @ frame, np.eye(frame.shape[1]), atol=1e-10):
            raise ValueError("orthonormal intervention frames required")
    d = y - x
    p = (d @ u) @ u.T
    if arm in ("full_property", "neutral"):
        return d
    if arm == "projection":
        return p
    if arm == "complement":
        return d - p
    if arm == "random_matched":
        z = (d @ r) @ r.T
        target, norm = np.linalg.norm(p, axis=1), np.linalg.norm(z, axis=1)
        if np.any((norm == 0) & (target > 0)):
            raise ValueError("random control cannot match nonzero displacement")
        return (
            z
            * np.divide(target, norm, out=np.zeros_like(target), where=norm > 0)[
                :, None
            ]
        )
    if arm == "noop":
        return np.zeros_like(d)
    raise ValueError("unknown fixed intervention arm")
