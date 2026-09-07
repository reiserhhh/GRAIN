#!/usr/bin/env python3
"""S12 representation collection checks on tiny synthetic data and random Qwen."""

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_query_representation import (
    capture_representations,
    state_summaries,
)


def test_equal_comment_not_token_weighted_and_query_excluded():
    h = np.arange(15, dtype=float).reshape(5, 3)
    body, query = state_summaries(h, [[0], [1, 2, 3]])
    np.testing.assert_allclose(body, (h[0] + h[1:4].mean(0)) / 2)
    np.testing.assert_array_equal(query, h[4])
    for bad in ([[0, 0]], [[0], [0]], [[4]], [[]]):
        with pytest.raises(ValueError):
            state_summaries(h, bad)


def test_single_forward_features_and_prefix_query_independence():
    torch.manual_seed(732)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    observer = DecoderObserver(model, None, max_tokens=64)
    alphabet = {"token_ids": [1, 2], "values": [0.0, 1.0]}
    first, second = [3, 5, 8, 11, 12, 9], [3, 5, 8, 11, 12, 10]
    _, a, error = capture_representations(
        observer, first, alphabet, [[0, 1], [2, 3]], block=0, layer=1
    )
    _, b, _ = capture_representations(
        observer, second, alphabet, [[0, 1], [2, 3]], block=0, layer=1
    )
    assert error < 1e-6
    assert all(value.shape == (32,) for value in a.values())
    np.testing.assert_allclose(a["body_mean"], b["body_mean"], atol=1e-7)
    assert not np.allclose(a["query_state"], b["query_state"])
    assert observer.calls == 2
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.layers[1].self_attn.o_proj._forward_pre_hooks
