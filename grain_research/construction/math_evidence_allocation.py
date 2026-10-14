#!/usr/bin/env python3
"""Declare evidence-bank composition without deleting local token selection."""

from __future__ import annotations

import numpy as np


def equal_banks(weights, start, end, second, prefix):
    """Fix two bank shares, retaining current-query outside and within-bank laws."""
    import torch

    if weights.ndim != 4 or not 0 < start < end < second < prefix <= weights.shape[-1]:
        raise ValueError("Two disjoint nonempty evidence banks and outside required")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Finite nonnegative attention weights required")
    x, z = weights[..., start:end], weights[..., second:prefix]
    mx, mz = x.sum(-1, keepdim=True), z.sum(-1, keepdim=True)
    if (mx <= 0).any() or (mz <= 0).any():
        raise ValueError("Unobserved/underflowed bank cannot acquire an imputed law")
    m = mx + mz
    changed = weights.clone()
    changed[..., start:end] = x / mx * (m / 2)
    changed[..., second:prefix] = z / mz * (m / 2)
    return changed


def observe(observer, cache, suffix, targets, address):
    """Apply ONE all-layer bank composition only while reading a frozen prefix."""
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    from grain_research.construction import math_behavior_predictive as B

    if observer.model.config._attn_implementation != "eager":
        raise ValueError("Inspected eager reader required")
    start, end, second, prefix = (
        address[k] for k in ("body_start", "first_end", "second_start", "prefix_length")
    )
    if cache.get_seq_length() != prefix:
        raise ValueError("Evidence-bank address must match the immutable cache")
    original, receipts = qwen.eager_attention_forward, []

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Reweight evidence units, not the header/seam/continuation or key values."""
        if key.shape[-2] != prefix + query.shape[-2] or kwargs.get("dropout", 0):
            raise ValueError("Full cached evaluation suffix with dropout0 required")
        _, weights = original(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )
        changed = equal_banks(weights, start, end, second, prefix)
        outside = torch.cat(
            (
                changed[..., :start] - weights[..., :start],
                changed[..., end:second] - weights[..., end:second],
                changed[..., prefix:] - weights[..., prefix:],
            ),
            -1,
        )
        x, z = changed[..., start:end], changed[..., second:prefix]
        mx, mz = weights[..., start:end].sum(-1), weights[..., second:prefix].sum(-1)
        receipts.append(
            {
                "layer": module.layer_idx,
                "outside_error": float(outside.abs().max()),
                "bank_balance_error": float((x.sum(-1) - z.sum(-1)).abs().max()),
                "row_mass_error": float(
                    (changed.sum(-1) - weights.sum(-1)).abs().max()
                ),
                "evidence_mass_min": float((mx + mz).min()),
                "evidence_mass_max": float((mx + mz).max()),
                "native_share_min": float((mx / (mx + mz)).min()),
                "native_share_max": float((mx / (mx + mz)).max()),
            }
        )
        output = torch.matmul(
            changed, qwen.repeat_kv(value, module.num_key_value_groups)
        )
        return output.transpose(1, 2).contiguous(), changed

    try:
        qwen.eager_attention_forward = interface
        rows = B.observe_candidates(observer, cache, suffix, targets)
    finally:
        qwen.eager_attention_forward = original
    if [r["layer"] for r in receipts] != list(range(len(observer.model.model.layers))):
        raise ValueError("Every reader layer must declare its bank composition")
    return rows, receipts


def split(native, declared):
    """Keep signed recursive allocation effects without positive-only attribution."""
    a, b = np.asarray(native, float), np.asarray(declared, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Complete aligned reader observations required")
    return {"native": a, "declared": b, "allocation_increment": a - b}
