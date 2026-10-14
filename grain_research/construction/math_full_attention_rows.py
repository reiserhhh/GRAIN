#!/usr/bin/env python3
"""Row-partitioned full attention for a fixed observation, not local attention."""

import torch

BACKEND = "grain_full_rows_hm227"


def full_attention_rows(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    """Keep every key and the original mask while partitioning query rows only.

    This deliberately supports only evaluation-time, uncached full histories.
    It does not expose attention weights or change the softmax key denominator.
    """
    rows = getattr(module.config, "grain_query_rows", None)
    if not isinstance(rows, int) or rows < 1:
        raise ValueError("A fixed positive query-row block is required")
    if module.training or dropout != 0 or kwargs.get("sliding_window") is not None:
        raise ValueError("Only fixed eval/full-attention observations are supported")
    if query.shape[-2] != key.shape[-2] or key.shape != value.shape:
        raise ValueError("No cache, partial key history, or changed K/V geometry")
    if (
        attention_mask is None
        or attention_mask.ndim != 4
        or attention_mask.dtype == torch.bool
        or attention_mask.shape[-2:] != (query.shape[-2], key.shape[-2])
    ):
        raise ValueError("Original full additive causal/padding mask required")
    groups = module.num_key_value_groups
    if query.shape[1] != key.shape[1] * groups:
        raise ValueError("Invalid grouped-query head binding")
    if groups != 1:
        batch, heads, length, dim = key.shape
        key = (
            key[:, :, None]
            .expand(batch, heads, groups, length, dim)
            .reshape(batch, heads * groups, length, dim)
        )
        value = (
            value[:, :, None]
            .expand(batch, heads, groups, length, dim)
            .reshape(batch, heads * groups, length, dim)
        )
    key_t = key.transpose(2, 3)
    outputs = []
    for start in range(0, query.shape[-2], rows):
        stop = min(start + rows, query.shape[-2])
        weights = torch.matmul(query[:, :, start:stop], key_t) * scaling
        weights = weights + attention_mask[:, :, start:stop]
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32)
        outputs.append(torch.matmul(weights.to(query.dtype), value))
        del weights
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None


def register():
    """Register attention AND the unchanged eager mask, without global defaults."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import eager_mask

    AttentionInterface.register(BACKEND, full_attention_rows)
    AttentionMaskInterface.register(BACKEND, eager_mask)


def install(model, rows):
    """Switch only this loaded frozen model after checking its full-context layout."""
    if not isinstance(rows, int) or rows < 1:
        raise ValueError("Fixed row block required")
    if model.config.model_type != "qwen2" or model.config.use_sliding_window:
        raise ValueError("Only the registered full-attention Qwen2 observer")
    register()
    model.config.grain_query_rows = rows
    model.set_attn_implementation(BACKEND)
    if model.config._attn_implementation != BACKEND:
        raise ValueError("Custom attention was not selected")
