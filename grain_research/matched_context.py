#!/usr/bin/env python3
"""Matched model-token observations and geometric pooling decomposition."""

from __future__ import annotations

import hashlib
import json

import numpy as np

ARMS = ("whole", "piece_equal", "piece_mass", "piece_mass_unit")


def id_hash(ids):
    """Commit to ordered model token IDs without revealing source text."""
    return hashlib.sha256(
        json.dumps(list(ids), separators=(",", ":")).encode()
    ).hexdigest()


def observations(text, tokenizer, budget=500, piece_size=128):
    """Create token-exact whole/piece inputs with original-character sidecars."""
    if budget < 1 or piece_size < 1:
        raise ValueError("Positive observation budgets required")
    raw = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = raw["input_ids"], raw["offset_mapping"]
    if not ids or len(ids) != len(offsets):
        raise ValueError("Nonempty source tokens and aligned offsets required")
    prefix = tokenizer.encode("query: ", add_special_tokens=False)
    ordinary = tokenizer.encode("query: " + text, add_special_tokens=False)
    if ordinary != prefix + ids:
        raise ValueError("Prefix changes source tokenization")
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    if tokenizer.encode("query: " + text, add_special_tokens=True) != [
        cls,
        *ordinary,
        sep,
    ]:
        raise ValueError("Unexpected E5 special-token template")
    n = min(budget, len(ids))
    ranges = [
        (0, n),
        *[(lo, min(lo + piece_size, n)) for lo in range(0, n, piece_size)],
    ]
    packets = []
    for k, (lo, hi) in enumerate(ranges):
        encoded = [cls, *prefix, *ids[lo:hi], sep]
        if len(encoded) > 512:
            raise ValueError("Explicit encoder cap exceeded")
        packets.append(
            {
                "kind": "whole" if k == 0 else "piece",
                "lo": lo,
                "hi": hi,
                "char_lo": offsets[lo][0],
                "char_hi": offsets[hi - 1][1],
                "ids": encoded,
                "content_sha256": id_hash(ids[lo:hi]),
            }
        )
    return packets, {
        "original_tokens": len(ids),
        "used_tokens": n,
        "omitted_tokens": len(ids) - n,
        "coverage": n / len(ids),
        "pieces": len(ranges) - 1,
        "complete_document": n == len(ids),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def pool_context(whole, pieces, masses):
    """Separate equal weighting, token mass and radial versus directional change."""
    whole, pieces, masses = [np.asarray(x, float) for x in (whole, pieces, masses)]
    if (
        whole.ndim != 1
        or pieces.ndim != 2
        or pieces.shape[1:] != whole.shape
        or masses.shape != (len(pieces),)
        or np.any(masses <= 0)
        or any(not np.isfinite(x).all() for x in (whole, pieces, masses))
    ):
        raise ValueError("Malformed whole/piece vectors or masses")
    if not np.allclose(np.linalg.norm(pieces, axis=1), 1, atol=1e-5) or not np.isclose(
        np.linalg.norm(whole), 1, atol=1e-5
    ):
        raise ValueError("Per-input vectors must be unit normalized")
    mean = (masses / masses.sum()) @ pieces
    radius = float(np.linalg.norm(mean))
    if radius < 1e-12:
        raise ValueError("Zero pooled direction: cannot normalize")
    unit = mean / radius
    direction, radial = whole - unit, (1 - radius) * unit
    np.testing.assert_allclose(whole - mean, direction + radial, atol=1e-12)
    return {
        "whole": whole,
        "piece_equal": pieces.mean(0),
        "piece_mass": mean,
        "piece_mass_unit": unit,
        "direction": direction,
        "radial": radial,
    }, radius
