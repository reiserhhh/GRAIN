#!/usr/bin/env python3
"""Forward-cut witnesses; no trained accuracy target or tolerance rescue."""

import numpy as np
import pytest

from grain_research.construction import math_query_relay_cut as M


def tiny_observer():
    """Build an untrained deterministic CPU network for interface tests only."""
    from types import SimpleNamespace

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(201)
    config = Qwen2Config(
        vocab_size=40,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=3,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return SimpleNamespace(
        model=Qwen2ForCausalLM(config).eval(),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )


def unit():
    """Use equal body content and padded unequal query lengths."""
    return {
        "input_ids": [[1, 2, 3, 4, 5] + [6 + q] * (q + 3) for q in range(5)],
        "body": [2, 3, 4],
    }


def test_support_and_exact_mask():
    import torch

    u = unit()
    r = M.relay_support(u)
    assert r[0] == [5, 6] and r[-1] == list(range(5, 11))
    mask = torch.zeros(5, 1, 12, 12)
    cut = M.cut_mask(mask, u["body"], r)
    assert mask.sum() == 0
    assert torch.isneginf(cut[0, 0, 5:7, 2:5]).all()
    assert (cut[0, 0, 7] == 0).all()
    with pytest.raises(ValueError, match="Contiguous"):
        M.relay_support({**u, "body": [2, 4]})
    with pytest.raises(ValueError, match="additive"):
        M.cut_mask(None, u["body"], r)


def test_factorial_signed_interactions():
    a = np.random.default_rng(201).normal(size=(7, 2, 2, 5, 3))
    f = M.factorial_actions(a)
    np.testing.assert_allclose(
        f["relay_delta"] + f["selection_delta"] + f["interaction"],
        f["joint_delta"],
        atol=1e-15,
    )
    a[..., 1, 1, :, :] = a[..., 1, 0, :, :] + a[..., 0, 1, :, :] - a[..., 0, 0, :, :]
    np.testing.assert_allclose(M.factorial_actions(a)["interaction"], 0, atol=1e-15)
    with pytest.raises(ValueError):
        M.factorial_actions(np.zeros((4, 5)))


def test_relay_is_content_blind_but_final_endpoint_is_not():
    """Changing same-length body leaves every cut relay state identical."""
    observer, u = tiny_observer(), unit()
    relay = M.relay_support(u)

    def capture(current, cut):
        saved, hooks = [], []

        def receive(_module, _args, output):
            state = output[0] if isinstance(output, tuple) else output
            saved.append(
                [state[q, rows].detach().numpy().copy() for q, rows in enumerate(relay)]
            )

        for layer in observer.model.model.layers:
            hooks.append(layer.register_forward_hook(receive))
        try:
            result = M.native_batch(observer, current, cut=cut)
        finally:
            for h in hooks:
                h.remove()
        return saved, result

    changed = {**u, "input_ids": [r[:2] + [23, 24, 25] + r[5:] for r in u["input_ids"]]}
    a, ra = capture(u, True)
    b, rb = capture(changed, True)
    for left, right in zip(a, b, strict=True):
        for x, y in zip(left, right, strict=True):
            np.testing.assert_array_equal(x, y)
    assert np.max(abs(ra[0] - rb[0])) > 1e-7
    an, _ = capture(u, False)
    bn, _ = capture(changed, False)
    assert (
        max(
            np.max(abs(x - y))
            for l, r in zip(an, bn, strict=True)
            for x, y in zip(l, r, strict=True)
        )
        > 1e-6
    )
    assert max(r["relay_body_weight_max"] for r in ra[5]) == 0


def test_foreign_fixed_c_self_and_restoration():
    import transformers.models.qwen2.modeling_qwen2 as qwen

    observer, u = tiny_observer(), unit()
    original = qwen.eager_attention_forward
    intact = M.native_batch(observer, u, cut=False)
    native = M.native_batch(observer, u)
    copied = intact[4].copy()
    foreign = M.native_batch(observer, u, [4, 2, 1, 3, 0], intact[4], fixed_c=True)
    identity = M.native_batch(observer, u, list(range(5)), native[4])
    np.testing.assert_allclose(identity[0], native[0], atol=3e-7, rtol=0)
    np.testing.assert_allclose(foreign[0][3], native[0][3], atol=3e-7, rtol=0)
    np.testing.assert_array_equal(copied, intact[4])
    assert np.max(abs(foreign[0][1] - native[0][1])) > 1e-9
    assert qwen.eager_attention_forward is original
    with pytest.raises(ValueError):
        M.native_batch(observer, u, [4, 2, 1, 0, 3], intact[4], fixed_c=True)
    assert qwen.eager_attention_forward is original


def test_global_hook_restored_after_forward_failure():
    import transformers.models.qwen2.modeling_qwen2 as qwen

    observer, u = tiny_observer(), unit()
    original = qwen.eager_attention_forward
    with pytest.raises((IndexError, ValueError)):
        M.native_batch(observer, u, range(5), np.ones((1, 5, 3, 3)))
    assert qwen.eager_attention_forward is original
