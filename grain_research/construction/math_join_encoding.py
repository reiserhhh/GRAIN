#!/usr/bin/env python3
"""Separate cross-event encoding from joint reading on unchanged evidence."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_late_condition as L


def isolate_mask(mask, length, body_start, first_end):
    """Close first-event keys into ALL subsequent prefix rows, including the seam."""
    import torch

    if not 0 < body_start < first_end < length:
        raise ValueError("A fixed header and two complete evidence blocks are required")
    if (
        mask is None
        or mask.ndim != 4
        or mask.shape[0] != 1
        or mask.shape[-2:] != (length, length)
    ):
        raise ValueError("Full single-prefix additive causal mask required")
    changed = mask.clone()
    changed[:, :, first_end:, body_start:first_end] = -torch.inf
    return changed


def encode(observer, prefix, body_start, first_end):
    """Keep tokens/positions fixed; isolate event encoders and restore the hook."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    if observer.model.config._attn_implementation != "eager":
        raise ValueError("Inspected eager attention required")
    length = len(prefix)
    if not 0 < body_start < first_end < length:
        raise ValueError("Two nonempty addressed evidence blocks required")
    original, receipts = qwen.eager_attention_forward, []

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Cut only encoding edges; subsequent behavioral reads remain native."""
        if query.shape[-2] != length or key.shape[-2] != length:
            raise ValueError("An encoding cut cannot silently act on suffix reading")
        changed = isolate_mask(attention_mask, length, body_start, first_end)
        output, weights = original(
            module, query, key, value, changed, scaling=scaling, **kwargs
        )
        closed = float(weights[:, :, first_end:, body_start:first_end].abs().max())
        if closed != 0:
            raise ArithmeticError("A cross-event encoding edge remained open")
        receipts.append({"layer": module.layer_idx, "closed_weight": closed})
        return output, weights

    try:
        qwen.eager_attention_forward = interface
        cache = L.encode_context(observer, prefix)
    finally:
        qwen.eager_attention_forward = original
    if [r["layer"] for r in receipts] != list(range(len(observer.model.model.layers))):
        raise ValueError("Every decoder layer must apply exactly one encoding cut")
    return cache, receipts


def split(native, isolated):
    """Keep independent-bank reading and its signed contextual-encoding increment."""
    a, b = np.asarray(native, float), np.asarray(isolated, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Aligned complete finite accounts required")
    return {"native": a, "isolated": b, "encoding_increment": a - b}
