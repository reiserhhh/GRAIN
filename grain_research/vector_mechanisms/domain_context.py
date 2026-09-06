#!/usr/bin/env python3
"""Frozen domain context materials and speaker-correct native token boundaries."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np


def salted(value):
    """Order candidates without semantic or psychological outcome selection."""
    return hashlib.sha256(f"vm06-domain-context-v1:{value}".encode()).hexdigest()


def sextile_spans(text):
    """Return six contiguous raw character spans with whitespace-aligned boundaries."""
    if not text or not text.strip():
        raise ValueError("Empty document")
    cuts = [0]
    for k in range(1, 6):
        p = len(text) * k // 6
        while p < len(text) and not text[p].isspace():
            p += 1
        if p <= cuts[-1] or p >= len(text):
            raise ValueError("Insufficient separate document sextiles")
        cuts.append(p)
    cuts.append(len(text))
    spans = [(cuts[i], cuts[i + 1]) for i in range(6)]
    if any(not text[a:b].strip() for a, b in spans):
        raise ValueError("Whitespace-only document sextile")
    if "".join(text[a:b] for a, b in spans) != text:
        raise ValueError("Nonlossless sextile design")
    return spans


def user_parts(tokenizer, current, context=None):
    """Isolate current and earlier user-text tokens without assigning an AI role."""
    if not current or not current.strip():
        raise ValueError("Empty current text")
    messages = [] if context is None else [{"role": "user", "content": context}]
    messages.append({"role": "user", "content": current})
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    start = rendered.rfind(current)
    if start < 0:
        raise ValueError("Current text absent from rendered template")
    stop = start + len(current)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    native = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if isinstance(native, Mapping):
        native = native["input_ids"]
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if list(ids) != list(native):
        raise ValueError("Native chat/offset tokenization disagree")

    def indices(a, b, strict):
        ix = [
            i
            for i, (left, right) in enumerate(offsets)
            if right > left >= a and right <= b
        ]
        if not ix or ix != list(range(ix[0], ix[-1] + 1)):
            raise ValueError("Noncontiguous content tokens")
        crossing = sum(
            right > a and left < b and (left < a or right > b)
            for left, right in offsets
            if right > left
        )
        if strict and crossing:
            raise ValueError("Current content crosses a template token boundary")
        return ix, crossing

    body_indices, _ = indices(start, stop, True)
    prefix, body = (
        list(ids[: body_indices[0]]),
        list(ids[body_indices[0] : body_indices[-1] + 1]),
    )
    if context is None:
        return prefix, body, [], {"context_boundary_crossing_tokens": 0}
    cstart = rendered.find(context, 0, start)
    if not context or cstart < 0 or rendered.find(context, cstart + 1, start) >= 0:
        raise ValueError("Ambiguous context text interval")
    context_indices, crossing = indices(cstart, cstart + len(context), False)
    if len(context_indices) < 16 or context_indices[-1] >= len(prefix):
        raise ValueError("Insufficient or overlapping context tokens")
    if {prefix[i] for i in context_indices} & set(tokenizer.all_special_ids):
        raise ValueError("Special token inside context")
    return (
        prefix,
        body,
        context_indices,
        {"context_boundary_crossing_tokens": int(crossing)},
    )


def qualify_case(case, tokenizer):
    """Keep complete current/context text or reject the whole candidate."""
    prefix, body, ix, extra = user_parts(tokenizer, case["current"], case["context"])
    _, empty_body, _, _ = user_parts(tokenizer, case["current"])
    length = len(tokenizer.encode(case["context"], add_special_tokens=False))
    if not 16 <= len(body) <= 256 or not 16 <= length <= 640 or len(prefix) > 768:
        raise ValueError("Complete candidate exceeds fixed token envelope")
    if body != empty_body:
        raise ValueError("Current tokens change with context presence")
    return case | {
        "current_tokens": len(body),
        "context_tokens": length,
        "prefix_tokens": len(prefix),
        "isolated_context_tokens": len(ix),
        "context_sha256": hashlib.sha256(case["context"].encode()).hexdigest(),
        "current_sha256": hashlib.sha256(case["current"].encode()).hexdigest(),
        **extra,
    }


def spread_indices(count):
    """Select six unique chronological indices; no nearest-date resampling."""
    if count < 6:
        raise ValueError("Need six distinct posts")
    indices = np.linspace(0, count - 1, 6, dtype=int)
    if len(set(indices)) != 6:
        raise ValueError("Repeated spread index")
    return indices
