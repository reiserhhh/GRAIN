#!/usr/bin/env python3
"""Replay Qwen suffixes from cached states and a fixed causal text partition."""

from __future__ import annotations

import numpy as np
import torch
from transformers.masking_utils import create_causal_mask


def query_region_mask(offsets, question_start):
    """Assign boundary-crossing tokens to the question region; exclude final query."""
    bounds = np.asarray(offsets)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or question_start <= 0:
        raise ValueError("Token offsets and a fixed question boundary required")
    mask = bounds[:, 1] > question_start
    mask[-1] = False
    if not mask.any() or mask[:-1].all():
        raise ValueError("Both evidence context and nonfinal question states required")
    return mask


def cached_suffix_kwargs(model, state):
    """Recreate native full-attention mask and RoPE from shape/positions, no prefix."""
    if model.config.model_type != "qwen2" or model.model.has_sliding_layers:
        raise ValueError("Only the pinned unwindowed Qwen2 observer is supported")
    positions = torch.arange(state.shape[1], device=state.device)[None]
    with torch.no_grad():
        mask = create_causal_mask(
            config=model.config,
            inputs_embeds=state,
            attention_mask=torch.ones_like(positions),
            past_key_values=None,
            position_ids=positions,
        )
        rope = model.model.rotary_emb(state, positions)
    return {
        "attention_mask": mask,
        "position_embeddings": rope,
        "position_ids": positions,
        "past_key_values": None,
        "use_cache": False,
    }


def split_context_state(aligned, edited, question_mask, arm):
    """Replace only E or P while keeping the final query at the aligned origin."""
    if (
        aligned.shape != edited.shape
        or len(question_mask) != aligned.shape[1]
        or arm not in ("E", "P")
    ):
        raise ValueError("Aligned states, mask and named region required")
    select = torch.as_tensor(question_mask, dtype=torch.bool, device=aligned.device)
    if bool(select[-1]):
        raise ValueError("Final query is not part of either context region")
    if arm == "E":
        select = ~select
        select[-1] = False
    result = aligned.clone()
    result[:, select] = edited[:, select]
    return result
