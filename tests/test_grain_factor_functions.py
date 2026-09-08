#!/usr/bin/env python3
"""Joint P/Q gradients checked against a tiny frozen transformer suffix."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.factor_functions import joint_gradient
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.query_functional import region_gradient


def test_joint_partial_derivatives_and_finite_action():
    torch.manual_seed(21)
    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=19,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    cfg._attn_implementation = "eager"
    model = Qwen2ForCausalLM(cfg).eval().double().requires_grad_(False)
    observer = SimpleNamespace(model=model, calls=0)
    h = torch.randn((1, 6, 16), dtype=torch.float64)
    mask = np.array([False, False, False, True, True, False])
    kw = cached_suffix_kwargs(model, h)
    alphabet = {"token_ids": [2, 7], "values": [0.0, 1.0], "labels": ["a", "b"]}
    score, G, g = joint_gradient(observer, h, mask, kw, alphabet)
    value, separate = region_gradient(observer, h, h[:, mask], mask, kw, alphabet)
    assert score == pytest.approx(value, abs=1e-7)
    np.testing.assert_allclose(G, separate, atol=1e-9)
    dp, dq = torch.randn_like(h[:, mask]), torch.randn_like(h[:, -1])
    step = 0.005
    vals = []
    for sign in (-1, 1):
        shifted = h.clone()
        shifted[:, mask] += sign * step * dp
        shifted[:, -1] += sign * step * dq
        vals.append(replay_suffix(observer, shifted, kw, alphabet)["conditional_score"])
    predicted = np.sum(G * dp[0].numpy()) + g @ dq[0].numpy()
    assert predicted == pytest.approx((vals[1] - vals[0]) / (2 * step), abs=2e-6)
    assert all(p.grad is None for p in model.parameters())
    mask[-1] = True
    with pytest.raises(ValueError, match="exclude"):
        joint_gradient(observer, h, mask, kw, alphabet)
