#!/usr/bin/env python3
"""Native-shaped tests for full-path JVPs and source-fixed readout construction."""

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.complete_transport import (
    body_direction,
    fixed_means,
    forward_response,
    post_attention,
    tail_contrast,
)
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs


def tiny_model(device="cpu"):
    """Use real pinned decoder operations with small untrained synthetic weights."""
    torch.manual_seed(41)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().to(device)
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_native_jvp_and_complete_residual_path(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; native run must not silently move devices")
    model = tiny_model(device)
    x = torch.randn(1, 9, 16, device=device).requires_grad_(True)
    kw = cached_suffix_kwargs(model, x)
    delta = torch.randn_like(x)
    delta[:, -2:] = 0
    ids = torch.tensor([2, 4, 6], device=device)
    c = torch.tensor([-0.5, 0.1, 0.4], device=device)
    r, direct, attention = post_attention(model, x, kw, start=0)
    torch.testing.assert_close(r, direct + attention)
    score = tail_contrast(model, r, ids, c)
    gradient, t = torch.autograd.grad(score, (x, r))
    with torch.no_grad():
        native = x.detach()
        for block in model.model.layers:
            native = block(native, **kw)
        native_score = (
            model.lm_head(model.model.norm(native[:, -1]))[:, ids] * c
        ).sum()
    torch.testing.assert_close(score, native_score, rtol=1e-5, atol=1e-7)
    primal, tangent = forward_response(model, x.detach(), kw, delta, start=0)
    torch.testing.assert_close(primal[0], r, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        tangent[0], tangent[1] + tangent[2], rtol=1e-5, atol=1e-7
    )
    torch.testing.assert_close(
        (tangent[0] * t).sum(), (gradient * delta).sum(), rtol=1e-4, atol=1e-7
    )
    assert tangent[1].norm() > 0  # Upstream BODY changes the final residual/query.
    eps = 0.001
    plus = post_attention(model, x.detach() + eps * delta, kw, 0)[0]
    minus = post_attention(model, x.detach() - eps * delta, kw, 0)[0]
    torch.testing.assert_close(
        (plus - minus) / (2 * eps), tangent[0], rtol=0.05, atol=1e-4
    )


def test_source_means_and_body_direction_are_explicit():
    rng = np.random.default_rng(41)
    source = {
        (i, q): np.ones(16) * (i + 10 * k)
        for i in range(4)
        for k, q in enumerate(("neuroticism", "confirmation_seeking"))
    }
    means = fixed_means(source, "neuroticism", held=3)
    np.testing.assert_allclose(means["own_t"], 1)
    np.testing.assert_allclose(means["pooled_t"], 6)
    x = torch.tensor(rng.normal(size=(1, 8, 16)), dtype=torch.float32)
    pos = np.arange(1, 6)
    d = body_direction(x, rng.normal(size=(5, 4)), pos, 0)
    assert d[:, 0].count_nonzero() == d[:, 6:].count_nonzero() == 0
    assert d[:, pos].norm() > 0
    with pytest.raises(ValueError, match="BODY"):
        body_direction(x, np.ones((2, 4)), np.array([1, 7]), 0)
