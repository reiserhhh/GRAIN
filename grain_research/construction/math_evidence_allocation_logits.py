#!/usr/bin/env python3
"""Factor the declared bank law from finite native logits, without zero filling."""

from __future__ import annotations


def conditional_banks(weights, logits, start, end, second, prefix):
    """Read conditional bank laws directly; native underflow is not structural absence."""
    import torch

    if (
        weights.shape != logits.shape
        or weights.ndim != 4
        or not 0 < start < end < second < prefix <= weights.shape[-1]
    ):
        raise ValueError("Aligned native laws/logits and two disjoint banks required")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Finite native nonnegative weights required")
    x, z = logits[..., start:end], logits[..., second:prefix]
    if not torch.isfinite(x).all() or not torch.isfinite(z).all():
        raise ValueError("All registered evidence keys must be structurally visible")
    mass = weights[..., start:end].sum(-1, keepdim=True) + weights[
        ..., second:prefix
    ].sum(-1, keepdim=True)
    changed = weights.clone()
    changed[..., start:end] = torch.softmax(x, -1, dtype=torch.float32).to(
        weights.dtype
    ) * (mass / 2)
    changed[..., second:prefix] = torch.softmax(z, -1, dtype=torch.float32).to(
        weights.dtype
    ) * (mass / 2)
    return changed


def observe(observer, cache, suffix, targets, address):
    """Use unchanged native current-query mass with logit-defined bank conditionals."""
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    from grain_research.construction import math_behavior_predictive as B

    if observer.model.config._attn_implementation != "eager":
        raise ValueError("Inspected eager attention required")
    start, end, second, prefix = (
        address[k] for k in ("body_start", "first_end", "second_start", "prefix_length")
    )
    if cache.get_seq_length() != prefix:
        raise ValueError("Immutable prefix address mismatch")
    original, receipts = qwen.eager_attention_forward, []

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Implement the same mathematical operation from observed pre-softmax inputs."""
        if key.shape[-2] != prefix + query.shape[-2] or kwargs.get("dropout", 0):
            raise ValueError("Full cached evaluation suffix required")
        _, weights = original(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )
        logits = (
            torch.matmul(
                query, qwen.repeat_kv(key, module.num_key_value_groups).transpose(2, 3)
            )
            * scaling
        )
        if attention_mask is not None:
            if (attention_mask[..., start:end] != 0).any() or (
                attention_mask[..., second:prefix] != 0
            ).any():
                raise ValueError(
                    "Registered evidence keys cannot be structurally masked"
                )
            logits = logits + attention_mask
        changed = conditional_banks(weights, logits, start, end, second, prefix)
        outside = torch.cat(
            (
                changed[..., :start] - weights[..., :start],
                changed[..., end:second] - weights[..., end:second],
                changed[..., prefix:] - weights[..., prefix:],
            ),
            -1,
        )
        mx, mz = weights[..., start:end].sum(-1), weights[..., second:prefix].sum(-1)
        receipts.append(
            {
                "layer": module.layer_idx,
                "outside_error": float(outside.abs().max()),
                "bank_balance_error": float(
                    (
                        changed[..., start:end].sum(-1)
                        - changed[..., second:prefix].sum(-1)
                    )
                    .abs()
                    .max()
                ),
                "row_mass_error": float(
                    (changed.sum(-1) - weights.sum(-1)).abs().max()
                ),
                "zero_native_first": int((mx == 0).sum()),
                "zero_native_second": int((mz == 0).sum()),
                "zero_native_combined": int(((mx + mz) == 0).sum()),
                "evidence_mass_min": float((mx + mz).min()),
                "evidence_mass_max": float((mx + mz).max()),
                "finite_bank_logits": True,
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
        raise ValueError("All reader layers must implement the declared operation")
    return rows, receipts
