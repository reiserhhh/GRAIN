#!/usr/bin/env python3
"""Local projection equals the real projected decoder prefix, with no suffix."""

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.configuration_mediation import projected_body
from grain_research.vector_mechanisms.configuration_source import (
    configuration_scores,
    full_head,
    projected_cache,
    replay_heads,
)
from grain_research.vector_mechanisms.prefix import capture_prefix
from grain_research.vector_mechanisms.trace import DecoderObserver


def test_actual_local_reprojection_matches_full_projected_prefix():
    torch.manual_seed(725)
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
    observer = DecoderObserver(Qwen2ForCausalLM(config).eval(), None, max_tokens=64)
    ids, groups = [3, 5, 8, 11, 12, 9], [[0, 1], [2, 3]]
    cache, trace = capture_prefix(observer, ids, block=0, layer=1)
    local, audit = projected_cache(observer, cache, groups, layer=1)
    assert observer.calls == 1
    assert trace["forbidden_calls"] == []
    with projected_body(observer.model.model.layers[0], groups):
        actual, actual_trace = capture_prefix(observer, ids, block=0, layer=1)
    for key in ("q", "k", "v"):
        np.testing.assert_allclose(local[key], actual[key], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(full_head(local), full_head(actual), atol=1e-7)
    assert audit == {"query_boundary_error": 0, "query_vector_error": 0}
    assert actual_trace["forbidden_calls"] == []
    assert not observer.model.model.layers[0]._forward_hooks
    with pytest.raises(ValueError, match="final query"):
        projected_cache(observer, cache, [[4, 5]], layer=1)


def test_grouped_heads_include_nonbody_tokens():
    cache = {
        "q": np.zeros((4, 2)),
        "k": np.zeros((3, 2, 2)),
        "v": np.arange(12, dtype=float).reshape(3, 2, 2),
    }
    expected = np.concatenate([cache["v"][:, h // 2].mean(0) for h in range(4)])
    np.testing.assert_allclose(full_head(cache), expected)
    with pytest.raises(ValueError, match="compatible"):
        full_head({**cache, "q": np.zeros((3, 2))})


def test_frozen_score_and_endpoint_replay():
    rng = np.random.default_rng(25)
    native, projected = rng.normal(size=(2, 3, 8))
    gradient = rng.normal(size=8)
    delta, scores, mean = configuration_scores(native, projected, gradient)
    np.testing.assert_allclose(delta, native - projected)
    np.testing.assert_allclose(mean, gradient @ delta.mean(0))
    assert (
        replay_heads(
            native[0], projected[0], native[0], projected[0], gradient, scores[0]
        )["linear_score_error"]
        < 1e-12
    )
    with pytest.raises(ValueError, match="replay"):
        replay_heads(
            native[0], projected[0], native[0] + 1, projected[0], gradient, scores[0]
        )
    with pytest.raises(ValueError, match="covector"):
        configuration_scores(native, projected, gradient[:3])
