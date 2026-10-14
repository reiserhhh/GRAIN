#!/usr/bin/env python3
"""Synthetic controls for target selection, not a new model accuracy gate."""

import numpy as np
import pytest

from grain_research.construction import math_target_selector_exchange as M


def fixture():
    """Return distinct query distributions with shared physical body support."""
    rng = np.random.default_rng(200)
    p = rng.uniform(0.1, 1, (5, 2, 8))
    p /= p.sum(-1, keepdims=True)
    body = np.array([1, 2, 3, 4])
    s = p[..., body] / p[..., body].sum(-1, keepdims=True)
    return p, body, s


def test_identity():
    p, body, s = fixture()
    np.testing.assert_allclose(M.redistribute(p, body, s, range(5)), p, atol=1e-16)


def test_body_mass_outside_and_fixed_c():
    p, body, s = fixture()
    changed = M.redistribute(p, body, s, [4, 2, 1, 3, 0])
    np.testing.assert_array_equal(changed[..., [0, 5, 6, 7]], p[..., [0, 5, 6, 7]])
    np.testing.assert_allclose(changed[..., body].sum(-1), p[..., body].sum(-1))
    np.testing.assert_allclose(changed[3], p[3], atol=1e-16)
    assert not np.allclose(changed[1], p[1])


def test_value_formula_and_constant_body_null():
    p, body, s = fixture()
    changed = M.redistribute(p, body, s, [4, 2, 1, 3, 0])
    values = np.random.default_rng(201).normal(size=(5, 2, 8, 3))
    actual = M.value_update(p, changed, values)
    direct = (changed[..., None] * values).sum(2) - (p[..., None] * values).sum(2)
    np.testing.assert_allclose(actual, direct, atol=3e-16)
    values[:, :, body] = values[:, :, 1:2]
    np.testing.assert_allclose(M.value_update(p, changed, values), 0, atol=3e-16)


@pytest.mark.parametrize("mapping", [[0, 0, 1, 2, 3], [0, 1], [0.0, 1, 2, 3, 4]])
def test_invalid_donor_assignment(mapping):
    with pytest.raises(ValueError):
        M.permutation(mapping, 5)


def test_unobserved_or_malformed_probability():
    p, body, s = fixture()
    with pytest.raises(ValueError):
        M.redistribute(p, body, s / 2, range(5))
    p[..., body] = 0
    with pytest.raises(ValueError):
        M.redistribute(p, body, s, range(5))


def test_signed_covariance_and_paired_null():
    x = np.arange(40, dtype=float).reshape(8, 5)
    np.testing.assert_allclose(M.covariances(x, -x), -np.cov(x, rowvar=False))
    np.testing.assert_array_equal(M.paired_diagonal_ci(x, x, x, 20, 200), 0)


def test_actual_eager_interface_identity_and_foreign_selection():
    """Exercise padded five-query routing in a tiny untrained CPU Qwen."""
    from types import SimpleNamespace

    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(200)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    observer = SimpleNamespace(
        model=Qwen2ForCausalLM(config).eval(),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    unit = {
        "input_ids": [[1, 2, 3, 4, 5] + [6 + q] * (q + 2) for q in range(5)],
        "body": [2, 3, 4],
    }
    original = qwen.eager_attention_forward
    logits, states, masses, receipts, donor = M.native_batch(observer, unit)
    assert qwen.eager_attention_forward is original
    assert donor.shape == (2, 5, 3, 3)
    assert masses.shape == (2, 5, 3)
    assert max(r["shared_body_KV_error"] for r in receipts) < 1e-6
    identity = M.native_batch(observer, unit, list(range(5)), donor)
    np.testing.assert_allclose(identity[0], logits, atol=2e-7, rtol=0)
    np.testing.assert_allclose(identity[1], states, atol=2e-6, rtol=0)
    foreign = M.native_batch(observer, unit, [4, 2, 1, 3, 0], donor)
    assert qwen.eager_attention_forward is original
    np.testing.assert_allclose(foreign[0][3], logits[3], atol=2e-7, rtol=0)
    assert np.max(abs(foreign[0][1] - logits[1])) > 1e-8
    assert np.mean([r["conditional_tv_mean"] for r in foreign[3]]) > 1e-6
    for r in foreign[3]:
        assert (
            max(r[k] for k in ("installed_error", "mass_error", "outside_error")) < 2e-6
        )


def test_invalid_native_support_precedes_model_call():
    """Reject malformed query support without a forward or global hook change."""
    from types import SimpleNamespace

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    config._attn_implementation = "eager"
    observer = SimpleNamespace(
        model=Qwen2ForCausalLM(config).eval(),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    with torch.inference_mode(), pytest.raises(ValueError, match="Five nonempty"):
        M.native_batch(observer, {"input_ids": [[1, 2]], "body": [0]})
    with pytest.raises(ValueError, match="explicit permutation"):
        M.native_batch(observer, {"input_ids": [[1, 2]] * 5, "body": [0]}, donor=[])
