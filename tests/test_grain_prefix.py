#!/usr/bin/env python3
"""Actual tiny-Qwen early exit, feature replay and reusable observer checks."""

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.prefix import capture_prefix, prefix_pairing
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_query_representation import (
    capture_representations,
    state_summaries,
)


def test_actual_prefix_stops_before_suffix_and_preserves_full_forward():
    torch.manual_seed(733)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    observer = DecoderObserver(model, None, max_tokens=64)
    ids, groups = [3, 5, 8, 11, 12, 9], [[0, 1], [2, 3]]
    alphabet = {"token_ids": [1, 2], "values": [0.0, 1.0]}
    score, full, _ = capture_representations(
        observer, ids, alphabet, groups, block=0, layer=1
    )
    cache, trace = capture_prefix(observer, ids, block=0, layer=1)
    body, query = state_summaries(cache["boundary"], groups)
    for name, value in (
        ("body_mean", body),
        ("query_state", query),
        ("kv_pairing", prefix_pairing(cache, groups)),
    ):
        np.testing.assert_allclose(value, full[name], rtol=1e-7, atol=1e-7)
    assert trace == {"completed_blocks": [0], "forbidden_calls": []}
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.layers[1].self_attn._forward_pre_hooks
    assert not model.lm_head._forward_pre_hooks
    again, _, _ = capture_representations(
        observer, ids, alphabet, groups, block=0, layer=1
    )
    assert again == pytest.approx(score, abs=1e-8)
    assert observer.calls == 3


def test_invalid_prefix_sites_fail_before_execution():
    class Empty:
        pass

    dummy = Empty()
    dummy.model = Empty()
    dummy.model.model = Empty()
    dummy.model.model.layers = [None, None]
    with pytest.raises(ValueError):
        capture_prefix(dummy, [1], block=0, layer=0)
