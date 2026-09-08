#!/usr/bin/env python3
"""Native cached-suffix arguments and disjoint context-region interventions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.context_carrier import (
    cached_suffix_kwargs,
    query_region_mask,
    split_context_state,
)
from grain_research.vector_mechanisms.full_state_edit import (
    capture_boundary,
    replay_suffix,
)


def test_question_partition_is_exhaustive_except_final():
    mask = query_region_mask([(0, 2), (2, 4), (4, 7), (7, 9), (9, 10)], 5)
    np.testing.assert_array_equal(mask, [False, False, True, True, False])
    a, b = torch.zeros((1, 5, 3)), torch.ones((1, 5, 3))
    e, p = split_context_state(a, b, mask, "E"), split_context_state(a, b, mask, "P")
    torch.testing.assert_close((e + p)[:, :-1], b[:, :-1])
    assert (e + p)[:, -1].sum() == 0


def test_reconstructed_suffix_arguments_match_native_and_run_no_prefix():
    torch.manual_seed(27)
    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=19,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    cfg._attn_implementation = "eager"
    model = Qwen2ForCausalLM(cfg).eval()
    observer = SimpleNamespace(model=model, calls=0)
    alphabet = {"token_ids": [2, 7], "values": [0.0, 1.0], "labels": ["a", "b"]}
    h, native, score = capture_boundary(observer, [3, 4, 5, 6, 7], alphabet)

    def forbidden(*_args):
        raise AssertionError("Cached suffix touched prefix")

    handle = model.model.layers[0].register_forward_pre_hook(forbidden)
    try:
        reconstructed = cached_suffix_kwargs(model, h)
        for name in ("position_ids", "attention_mask"):
            torch.testing.assert_close(reconstructed[name], native[name])
        for a, b in zip(
            reconstructed["position_embeddings"],
            native["position_embeddings"],
            strict=True,
        ):
            torch.testing.assert_close(a, b)
        result = replay_suffix(observer, h, reconstructed, alphabet)
    finally:
        handle.remove()
    assert result["conditional_score"] == pytest.approx(
        score["conditional_score"], abs=1e-7
    )
