#!/usr/bin/env python3
"""Distinct native mechanisms, complete character edits, and safe value surgery."""

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.construction import math_joint_reading as J
from grain_research.construction import math_native_joint_value as M


def values():
    """Return toy physical GQA corners, not neural or human observations."""
    return np.random.default_rng(11).normal(size=(4, 5, 2, 4))


def test_only_joint_sector_removed_and_random_head_norms():
    v = values()
    delta, random = M.joint_edits(v, "fixed")
    before, after = J.coefficients(v), J.coefficients(v + delta)
    np.testing.assert_allclose(before[:3], after[:3], atol=1e-14)
    np.testing.assert_allclose(after[3], 0, atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.norm(delta, axis=(1, 3)),
        np.linalg.norm(random, axis=(1, 3)),
        atol=1e-14,
    )
    np.testing.assert_array_equal(random, M.joint_edits(v, "fixed")[1])
    assert not np.allclose(random, M.joint_edits(v, "other")[1])


def test_native_current_carried_signal_vs_native_bypass():
    v = values()
    delta, _ = M.joint_edits(v, "case")
    local = v.sum(axis=(1, 2, 3))
    native_current = local + 2
    native_patched = (v + delta).sum(axis=(1, 2, 3)) + 2
    np.testing.assert_allclose(
        J.coefficients(native_current - native_patched)[3], J.coefficients(local)[3]
    )
    native_bypass = 7 * J.BASIS[3]
    assert J.coefficients(native_bypass)[3] == 7
    np.testing.assert_allclose(J.coefficients(native_bypass - native_bypass), 0)
    np.testing.assert_allclose(J.coefficients(np.ones(4))[1:], 0)


def test_author_law_preserves_all_native_units_and_cancellation():
    f = np.random.default_rng(9).normal(size=(4, 6, 4))
    plus, minus = (1 + 0.5 * J.BASIS[3]) / 4, (1 - 0.5 * J.BASIS[3]) / 4
    np.testing.assert_allclose(
        np.tensordot(plus - minus, f, axes=1), J.coefficients(f)[3]
    )
    loss = 2 * f
    np.testing.assert_allclose(
        J.coefficients(f), J.coefficients(loss) + J.coefficients(f - loss)
    )


def test_native_units_shift_and_saturation():
    z = np.arange(10, dtype=float) * 20
    a = M.native_units(z, logsumexp(z) + 1)
    b = M.native_units(z + 9, logsumexp(z) + 10)
    np.testing.assert_allclose(a, b)
    assert a[0] > 0.9999 and a[1] > 90
    assert abs(a[2] - np.exp(-1)) < 1e-12
    with pytest.raises(ValueError):
        M.native_units(z[:-1], 3)
    with pytest.raises(ValueError):
        M.joint_edits(np.full((4, 5, 2, 4), np.nan), "invalid")


def test_value_edit_touches_only_declared_rows():
    import torch

    original = torch.arange(28, dtype=torch.float32).reshape(1, 7, 4)
    delta = np.ones((2, 4))
    updated = M.edit_values(original, [2, 4], delta)
    np.testing.assert_array_equal(
        updated[0, [0, 1, 3, 5, 6]], original[0, [0, 1, 3, 5, 6]]
    )
    np.testing.assert_array_equal(updated[0, [2, 4]], original[0, [2, 4]] + 1)
    assert torch.equal(original, torch.arange(28, dtype=torch.float32).reshape(1, 7, 4))
    with pytest.raises(ValueError):
        M.edit_values(original, [2, 2], delta)
    with pytest.raises(ValueError):
        M.edit_values(original, [2, 4], np.ones((2, 5)))


def test_scoped_native_hook_mask_and_cleanup_on_failure():
    import types

    import torch

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.k_proj = torch.nn.Linear(2, 2, bias=False)
            self.v_proj = torch.nn.Linear(2, 2, bias=False)
            self.fail = False

        def forward(self, hidden, attention_mask, position_embeddings):
            self.k_proj(hidden)
            if self.fail:
                raise RuntimeError("deliberate failure")
            return self.v_proj(hidden)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(12, 2)
            self.model = torch.nn.Module()
            block = torch.nn.Module()
            block.self_attn = Attention()
            self.model.layers = torch.nn.ModuleList([block])

        def forward(self, input_ids, **kwargs):
            hidden = self.emb(input_ids)
            n = input_ids.shape[1]
            mask = torch.zeros((1, 1, n, n)).masked_fill(
                ~torch.tril(torch.ones((n, n), dtype=torch.bool)),
                torch.finfo(torch.float32).min,
            )
            pos = (torch.ones((1, n, 2)), torch.zeros((1, n, 2)))
            v = self.model.layers[0].self_attn(
                hidden, attention_mask=mask, position_embeddings=pos
            )
            logits = v.sum() * torch.arange(10) / 20
            return types.SimpleNamespace(logits=logits[None, None])

    model = Model().eval()
    observer = types.SimpleNamespace(model=model, calls=0, max_tokens=20)
    row = {"input_ids": [1, 2, 3, 4], "current": [2, 3], "padding": [1]}
    alphabet = {
        "labels": [str(i) for i in range(10)],
        "values": (np.arange(10) / 9).tolist(),
        "token_ids": list(range(10)),
    }
    _, captured = M.native_forward(
        observer, row, [5], alphabet, np.ones((2, 1, 2)), layer=0
    )
    np.testing.assert_allclose(captured["after"] - captured["before"], 1, atol=1e-6)
    np.testing.assert_array_equal(captured["masked_blocks"], [0])
    attn = model.model.layers[0].self_attn
    assert (
        not attn._forward_pre_hooks
        and not attn.v_proj._forward_hooks
        and not attn.k_proj._forward_hooks
    )
    attn.fail = True
    with pytest.raises(RuntimeError, match="deliberate failure"):
        M.native_forward(observer, row, [5], alphabet, np.ones((2, 1, 2)), layer=0)
    assert (
        not attn._forward_pre_hooks
        and not attn.v_proj._forward_hooks
        and not attn.k_proj._forward_hooks
    )
