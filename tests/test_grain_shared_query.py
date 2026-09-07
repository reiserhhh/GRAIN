#!/usr/bin/env python3
"""Frozen per-view query projection and train-only reference construction."""

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_shared_query import reference_queries


def test_query_reference_matches_fixed_model_transform_and_excludes_fold():
    torch.manual_seed(734)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).eval()
    observer = DecoderObserver(model, None)
    states = np.random.default_rng(734).normal(size=(6, 3, 32)).astype(np.float32)
    folds = np.array([0, 1, 2, 0, 1, 2])
    refs = reference_queries(observer, states, folds, layer_index=1)
    layer = model.model.layers[1]
    with torch.no_grad():
        raw = layer.self_attn.q_proj(layer.input_layernorm(torch.from_numpy(states)))
    expected = raw.numpy().reshape(6, 3, 4, 8)[folds != 1].mean(axis=(0, 1))
    np.testing.assert_allclose(refs[1], expected, atol=1e-7)
    states[folds == 1] *= -3
    changed = reference_queries(observer, states, folds, layer_index=1)
    np.testing.assert_array_equal(refs[1], changed[1])
