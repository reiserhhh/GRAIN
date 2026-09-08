#!/usr/bin/env python3
"""Tiny-model upstream differentiation, real intervention and hook cleanup."""

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.trace import DecoderObserver
from grain_research.vector_mechanisms.upstream_routes import (
    assemble_patch,
    capture_differentiable_prefix,
    rank_route_candidates,
)


def fixture():
    """Construct a deterministic local toy decoder, without downloads or GPU."""
    torch.manual_seed(70)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().double().requires_grad_(False)
    observer = DecoderObserver(model, None, max_tokens=32)
    rng = np.random.default_rng(7)
    f = FunctionalQuery(
        np.zeros((2, 16)),
        np.zeros(16),
        rng.normal(size=(2, 16)),
        rng.normal(size=16),
        0.5,
    )
    return observer, f


def test_differentiation_patch_finite_difference_and_cleanup():
    obs, f = fixture()
    ids, other, mask = (
        [1, 3, 6, 9, 2],
        [1, 4, 6, 9, 2],
        [False, False, True, True, False],
    )
    r = capture_differentiable_prefix(obs, ids, mask, f, block=1)
    d = capture_differentiable_prefix(obs, other, mask, f, block=1, gradients=False)
    assert r["score"] == pytest.approx(f.parts(r["P"], r["Q"])["score"], abs=1e-7)
    assert r["trace"] == [0, 1] and np.isfinite(r["input_gradient"]).all()
    candidates = rank_route_candidates(r, d, 8)
    candidate = next(c for c in candidates if c["site"] == "attn:0")
    values = []
    for alpha in (-0.01, 0.01):
        patched = capture_differentiable_prefix(
            obs,
            ids,
            mask,
            f,
            block=1,
            gradients=False,
            patches=assemble_patch(r, d, [candidate], 8, alpha=alpha),
        )
        values.append(patched["score"])
    assert (values[1] - values[0]) / 0.02 == pytest.approx(
        candidate["effect_estimate"], abs=2e-6
    )
    same = capture_differentiable_prefix(
        obs,
        ids,
        mask,
        f,
        block=1,
        gradients=False,
        patches=assemble_patch(r, r, [candidate], 8),
    )
    assert same["score"] == pytest.approx(r["score"], abs=1e-7)
    with pytest.raises(ValueError, match="shape"):
        capture_differentiable_prefix(
            obs, ids, mask, f, block=1, patches={"mlp:0": np.ones((1, 1))}
        )
    for layer in obs.model.model.layers:
        assert (
            not layer._forward_hooks
            and not layer.mlp._forward_hooks
            and not layer.self_attn.o_proj._forward_pre_hooks
        )
    assert all(p.grad is None and not p.requires_grad for p in obs.model.parameters())


def test_native_patch_and_invalid_site():
    obs, f = fixture()
    args = (obs, [1, 3, 6, 9, 2], [False, False, True, True, False], f)
    alphabet = {"token_ids": [1, 2], "labels": ["0", "1"], "values": [0.0, 1.0]}
    result = capture_differentiable_prefix(
        *args, block=1, gradients=False, alphabet=alphabet
    )
    assert 0 <= result["native"]["conditional_score"] <= 1
    with pytest.raises(ValueError, match="Unknown"):
        capture_differentiable_prefix(
            *args, block=1, patches={"mlp:2": np.zeros((3, 16))}
        )
