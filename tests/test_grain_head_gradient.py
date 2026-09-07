#!/usr/bin/env python3
"""Check actual head-hook autodiff, path completeness and author-out prediction."""

import numpy as np
import pytest
import torch

from grain_research.vector_mechanisms.head_gradient import (
    leave_author_out_geometry,
    projection_gradient,
    simpson_gradient,
)


@pytest.mark.parametrize("quadratic", [False, True])
def test_leaf_hook_linear_quadratic_path_and_cleanup(quadratic):
    projection = torch.nn.Linear(3, 3, bias=False).double()
    with torch.no_grad():
        projection.weight.copy_(torch.eye(3, dtype=torch.float64))
    projection.requires_grad_(False)
    h = torch.tensor([1.0, -0.2, 0.7], dtype=torch.float64)
    d = torch.tensor([0.3, 0.1, -0.4], dtype=torch.float64)
    state = torch.stack([h * 3, h])[None]
    beta = torch.tensor([0.5, -0.7, 0.2], dtype=torch.float64)

    def forward():
        x = projection(state)[0, -1]
        return x @ beta + (0.5 * (x @ x) if quadratic else 0)

    values, gradients = {}, {}
    for t in (0.0, 0.5, 1.0):
        values[t], gradients[t] = projection_gradient(projection, h - t * d, forward)
        np.testing.assert_allclose(
            gradients[t], (beta + (h - t * d if quadratic else 0)).numpy(), atol=1e-12
        )
        assert not projection._forward_pre_hooks
        assert all(
            p.grad is None and not p.requires_grad for p in projection.parameters()
        )
    integrated = simpson_gradient(gradients)
    assert integrated @ d.numpy() == pytest.approx(values[0.0] - values[1.0], abs=1e-12)
    assert values[0.0] == pytest.approx(float(forward()))

    def broken():
        projection(state)
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        projection_gradient(projection, h, broken)
    assert not projection._forward_pre_hooks


def test_loo_prediction_does_not_use_held_out_gradient():
    rng = np.random.default_rng(19)
    gradients, d = rng.normal(size=(8, 6)), rng.normal(size=(8, 6))
    rows = leave_author_out_geometry(gradients, d)
    changed = gradients.copy()
    changed[0] *= 1000
    other = leave_author_out_geometry(changed, d)
    assert rows[0]["loo_mean_prediction"] == other[0]["loo_mean_prediction"]
    assert rows[0]["self_integral_prediction"] != other[0]["self_integral_prediction"]
    common = np.tile(gradients[0], (8, 1))
    rows = leave_author_out_geometry(common, d)
    for row in rows:
        assert row["loo_mean_prediction"] == pytest.approx(
            row["self_integral_prediction"]
        )
        assert row["rank1_gradient_error"] < 1e-28


def test_five_node_simpson_and_invalid_inputs():
    nodes = {t: np.array([t**3, 1.0]) for t in (0.0, 0.25, 0.5, 0.75, 1.0)}
    np.testing.assert_allclose(simpson_gradient(nodes), [0.25, 1.0])
    with pytest.raises(ValueError, match="Simpson"):
        simpson_gradient({0.0: np.ones(3), 1.0: np.ones(3)})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_gradient_moves_to_cpu_before_float64_export():
    projection = torch.nn.Linear(3, 3, bias=False).to("mps").requires_grad_(False)
    donor = torch.ones(3, device="mps")
    state = torch.ones((1, 2, 3), device="mps")
    value, gradient = projection_gradient(
        projection, donor, lambda: projection(state)[0, -1].square().sum()
    )
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert gradient.dtype == np.float64
    assert not projection._forward_pre_hooks


def test_tiny_qwen_native_gradient_replay_and_finite_difference():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from grain_research.vector_mechanisms.trace import DecoderObserver
    from scripts.run_grain_attention_pairing import capture_head, patch_head_score
    from scripts.run_grain_multiquery_sensitivity import conditional_torch

    torch.manual_seed(190)
    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    model = Qwen2ForCausalLM(cfg).eval().double().requires_grad_(False)
    observer = DecoderObserver(model, None, max_tokens=64)
    ids, alphabet = [3, 5, 8, 11, 9], {"token_ids": [1, 2], "values": [0.0, 1.0]}
    score, cache = capture_head(observer, ids, alphabet, layer=1)
    donor = torch.tensor(cache["native_head"].reshape(-1), dtype=torch.float64)
    tokens = torch.tensor([ids])

    def forward():
        logits = model(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        ).logits[0, -1]
        return conditional_torch(logits, alphabet)

    value, gradient = projection_gradient(
        model.model.layers[1].self_attn.o_proj, donor, forward
    )
    assert value == pytest.approx(score, abs=1e-7)
    direction = np.random.default_rng(190).normal(size=32)
    epsilon = 1e-3
    plus = patch_head_score(
        observer, ids, alphabet, donor.numpy() + epsilon * direction, layer=1
    )
    minus = patch_head_score(
        observer, ids, alphabet, donor.numpy() - epsilon * direction, layer=1
    )
    assert gradient @ direction == pytest.approx(
        (plus - minus) / (2 * epsilon), abs=1e-4
    )
    assert all(p.grad is None for p in model.parameters())
    assert not model.model.layers[1].self_attn.o_proj._forward_pre_hooks
