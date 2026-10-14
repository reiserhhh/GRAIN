#!/usr/bin/env python3
"""Synthetic algebra and actual tiny-decoder tests of the crossed interface."""

from types import SimpleNamespace

import numpy as np
import pytest

from grain_research.construction import math_target_crossed_evidence as M


def test_complete_accounts_keep_signed_components():
    rng = np.random.default_rng(232)
    raw = rng.normal(size=(32, 2, 6, 5))
    a = M.crossed_accounts(raw)
    np.testing.assert_allclose(
        a["matrix"], a["grand"] + a["reader"] + a["source"] + a["interaction"]
    )
    np.testing.assert_allclose(a["interaction"].sum(1), 0, atol=1e-14)
    np.testing.assert_allclose(a["interaction"].sum(2), 0, atol=1e-14)
    np.testing.assert_allclose(raw[:, 0], a["odd"] + a["even"])
    np.testing.assert_allclose(raw[:, 1], a["even"] - a["odd"])


def test_source_blind_null_is_zero_without_forcing_reader_equality():
    rng = np.random.default_rng(5)
    raw = np.broadcast_to(rng.normal(size=(3, 2, 1, 5)), (3, 2, 6, 5)).copy()
    a = M.crossed_accounts(raw)
    np.testing.assert_array_equal(a["departure"], np.zeros((3, 5, 5)))
    np.testing.assert_allclose(a["interaction"], 0, atol=1e-15)
    assert np.linalg.norm(a["reader"]) > 0


def test_crossed_interaction_can_exist_without_any_diagonal_advantage():
    raw = np.zeros((1, 2, 6, 5))
    raw[0, 0, 0, 1] = 1
    a = M.crossed_accounts(raw)
    assert np.linalg.norm(a["interaction"]) > 0
    assert np.trace(a["matrix"][0]) == 0


def test_common_specific_are_not_discarded():
    raw = np.random.default_rng(12).normal(size=(2, 5, 1, 3, 4))
    common, specific = M.common_and_specific(raw)
    np.testing.assert_allclose(raw, common + specific)
    np.testing.assert_allclose(specific.mean(1), 0, atol=1e-15)


def test_token_suffix_identity_required():
    rows = [[1, q + 2, 7, 8, 9] for q in range(5)]
    assert M.suffix_support(rows, 2) == [[2, 3]] * 5
    rows[-1][-2] = 6
    with pytest.raises(ValueError, match="identity"):
        M.suffix_support(rows, 2)


def test_tiny_neural_capture_import_and_closure():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(232)
    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    model.set_attn_implementation("eager")
    observer = SimpleNamespace(model=model, tokenizer=SimpleNamespace(pad_token_id=0))
    rows = [[1, 2, 3, 4, 5 + q, 11, 12, 13] for q in range(5)]
    unit = {"input_ids": rows, "body": [1, 2, 3]}
    native, _, v, _ = M.observe(observer, unit, 2)
    with torch.inference_mode():
        ref = model(
            torch.tensor(rows), attention_mask=torch.ones(5, 8), use_cache=False
        ).logits[:, -1]
    np.testing.assert_allclose(native, ref.numpy(), atol=1e-7)
    _, _, closed, receipts = M.observe(observer, unit, 2, closed=True)
    assert all(r["closed_body_access"] == 0 for r in receipts)
    common, _ = M.common_and_specific(v - closed)
    donor = np.broadcast_to(common, v.shape).copy()
    zero, _, _, baseline = M.observe(observer, unit, 2, donor=np.zeros_like(v))
    np.testing.assert_allclose(zero, native, atol=1e-7)
    imported, _, _, touched = M.observe(observer, unit, 2, donor=donor)
    assert [r["prefix_KV_hash"] for r in touched] == [
        r["prefix_KV_hash"] for r in baseline
    ]
    assert np.linalg.norm(imported - native) > 0
