#!/usr/bin/env python3
"""Finite integration, reference isolation and actual differentiable suffix tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.query_functional import (
    frozen_reference_matrix,
    hidden_subspace_overlap,
    integrated_matrix,
    matrix_response,
    matrix_variants,
    region_gradient,
    response_quality,
)


def test_simpson_recovers_two_condition_cubic_paths():
    z0, d = np.array([[1.0, 2.0], [-1.0, 0.5]]), np.array([[0.4, -0.3], [0.7, 0.8]])
    gs = [[3 * (z0 + t * d) ** 2 + e for t in (0, 0.5, 1)] for e in (0, 1)]
    predicted = matrix_response(integrated_matrix(gs), d)
    expected = np.sum((z0 + d) ** 3 - z0**3 + 0.5 * d)
    assert predicted == pytest.approx(expected)


def test_evaluation_matrices_do_not_enter_frozen_readout():
    g = np.arange(48.0).reshape(4, 3, 4)
    roles = ["reference", "evaluation", "reference", "evaluation"]
    ref = frozen_reference_matrix(g, roles)
    g[[1, 3]] = -1e9
    np.testing.assert_array_equal(frozen_reference_matrix(g, roles), ref)
    with pytest.raises(ValueError):
        frozen_reference_matrix(g, ["reference"] * 4)


def test_token_mean_can_erase_nonzero_function_without_erasing_hidden_energy():
    g = np.array([[1.0, 0.0], [-1.0, 0.0]])
    variants, info = matrix_variants(g)
    assert matrix_response(variants["frozen_full"], g) == 2
    assert matrix_response(variants["token_mean"], g) == 0
    assert info["rank1_energy"] == pytest.approx(1)
    assert info["rank95"] == 1
    assert matrix_response(variants["rank1"], g) == pytest.approx(2)


def test_hidden_overlap_handles_different_token_counts():
    a = np.diag([3.0, 2.0, 1.0, 0.5])
    b = np.vstack([a, a])
    assert hidden_subspace_overlap(a, b, rank=2) == pytest.approx(1)
    assert hidden_subspace_overlap(a[:2], a[2:], rank=2) == pytest.approx(0)


def test_two_author_quality_is_error_not_correlation():
    metrics = response_quality([0.1, -0.2], [0.08, -0.1])
    assert metrics["same_sign"] == 2
    assert metrics["error_energy_over_zero"] == pytest.approx(0.208)
    assert "pearson" not in metrics


def test_region_gradient_matches_finite_difference_without_prefix_or_weight_grad():
    torch.manual_seed(71)
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

    def forbidden(*_args):
        raise AssertionError("Prefix was called")

    hook = model.model.layers[0].register_forward_pre_hook(forbidden)
    try:
        score, gradient = region_gradient(observer, h, h[:, mask], mask, kw, alphabet)
        native = replay_suffix(observer, h, kw, alphabet)["conditional_score"]
        direction = torch.randn_like(h[:, mask])
        values = []
        for sign in (-1, 1):
            shifted = h.clone()
            shifted[:, mask] += sign * 0.01 * direction
            values.append(
                replay_suffix(observer, shifted, kw, alphabet)["conditional_score"]
            )
    finally:
        hook.remove()
    expected = (values[1] - values[0]) / 0.02
    assert score == pytest.approx(native, abs=1e-7)
    assert matrix_response(gradient, direction[0].numpy()) == pytest.approx(
        expected, abs=2e-6
    )
    assert all(p.grad is None for p in model.parameters())
