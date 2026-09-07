#!/usr/bin/env python3
"""Real architecture smoke tests using a tiny random CPU model, not source data."""

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_attention_pairing import (
    analyze_heads,
    capture_head,
    patch_head_score,
    rotate_half,
)


def tiny_observer():
    torch.manual_seed(731)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    return DecoderObserver(model, None, max_tokens=64)


def test_capture_reconstruct_and_native_copy_on_tiny_qwen():
    observer = tiny_observer()
    ids, alphabet = [3, 5, 8, 11, 12, 9], {"token_ids": [1, 2], "values": [0.0, 1.0]}
    origin, cache = capture_head(observer, ids, alphabet, layer=0)
    rows, donor = analyze_heads(cache, [[0, 1], [2, 3]])
    assert len(rows) == 4
    assert max(r["reconstruction_max_error"] for r in rows) < 1e-6
    copied = patch_head_score(observer, ids, alphabet, cache["native_head"], layer=0)
    assert copied == pytest.approx(origin, abs=1e-7)
    changed = patch_head_score(observer, ids, alphabet, donor, layer=0)
    assert np.isfinite(changed)
    assert observer.calls == 3
    attention = observer.model.model.layers[0].self_attn
    assert not attention._forward_pre_hooks
    assert not attention.o_proj._forward_pre_hooks
    assert not attention.q_proj._forward_hooks


def test_rotary_half_map_squares_to_negative_identity():
    x = np.arange(24).reshape(3, 8)
    assert np.array_equal(rotate_half(rotate_half(x)), -x)
    with pytest.raises(ValueError, match="even"):
        rotate_half(np.zeros(3))
