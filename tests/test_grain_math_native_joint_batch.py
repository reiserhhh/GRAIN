#!/usr/bin/env python3
"""Same-shape batched endpoints, native calibration and hook safety."""

import types

import numpy as np
import pytest
import torch

from grain_research.construction.math_native_joint_batch import native_batch


class Attention(torch.nn.Module):
    """Small deterministic projection, not a language mechanism experiment."""

    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.k_proj = torch.nn.Linear(2, 2, bias=False)
        self.v_proj = torch.nn.Linear(2, 2, bias=False)
        self.fail = False

    def forward(self, hidden, **kwargs):
        self.k_proj(hidden)
        if self.fail:
            raise RuntimeError("deliberate batch failure")
        return self.v_proj(hidden)


class Model(torch.nn.Module):
    """Expose true endpoint-dependent logits under key-only pad permissions."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.emb = torch.nn.Embedding(15, 2)
        self.model = torch.nn.Module()
        block = torch.nn.Module()
        block.self_attn = Attention()
        self.model.layers = torch.nn.ModuleList([block])

    def forward(
        self, input_ids, attention_mask, position_ids, logits_to_keep, **kwargs
    ):
        b, n = input_ids.shape
        allowed = (
            torch.tril(torch.ones((n, n), dtype=torch.bool))[None, None]
            & attention_mask[:, None, None].bool()
        )
        mask = torch.zeros((b, 1, n, n)).masked_fill(
            ~allowed, torch.finfo(torch.float32).min
        )
        hidden = self.emb(input_ids)
        pos = (torch.ones((b, n, 2)), torch.zeros((b, n, 2)))
        value = self.model.layers[0].self_attn(
            hidden, attention_mask=mask, position_embeddings=pos
        )
        running = value.cumsum(1).sum(-1)
        logits = running[:, logits_to_keep, None] * torch.arange(10)[None, None] / 20
        return types.SimpleNamespace(logits=logits)


def setup():
    """Use unequal suffix lengths to expose an accidental padded-endpoint read."""
    model = Model().eval()
    observer = types.SimpleNamespace(model=model, calls=0, max_tokens=25)
    row = {"input_ids": [1, 2, 3, 4], "current": [2, 3], "padding": [1]}
    suffixes = [[5], [6, 7, 8], [9, 10]]
    alphabet = {
        "labels": [str(i) for i in range(10)],
        "values": (np.arange(10) / 9).tolist(),
        "token_ids": list(range(10)),
    }
    return observer, row, suffixes, alphabet


def test_same_current_different_true_endpoints():
    obs, row, suffixes, alphabet = setup()
    scores, raw = native_batch(obs, row, suffixes, alphabet, None, layer=0)
    np.testing.assert_array_equal(raw["endpoint"], [4, 6, 5])
    np.testing.assert_array_equal(raw["execution_length"], [7, 7, 7])
    np.testing.assert_array_equal(raw["before"][0], raw["before"][1])
    np.testing.assert_array_equal(raw["k"][0], raw["k"][2])
    assert len({r["conditional_score"] for r in scores}) == 3
    assert not np.array_equal(
        raw["execution_support_hash"][0], raw["execution_support_hash"][1]
    )
    assert obs.calls == 1


def test_native_edit_and_cleanup_on_actual_exception():
    obs, row, suffixes, alphabet = setup()
    _, raw = native_batch(obs, row, suffixes, alphabet, np.ones((2, 1, 2)), layer=0)
    np.testing.assert_allclose(raw["after"] - raw["before"], 1, atol=1e-6)
    attn = obs.model.model.layers[0].self_attn
    assert (
        not attn._forward_pre_hooks
        and not attn.k_proj._forward_hooks
        and not attn.v_proj._forward_hooks
    )
    attn.fail = True
    with pytest.raises(RuntimeError, match="deliberate batch failure"):
        native_batch(obs, row, suffixes, alphabet, None, layer=0)
    assert (
        not attn._forward_pre_hooks
        and not attn.k_proj._forward_hooks
        and not attn.v_proj._forward_hooks
    )


def test_invalid_value_support_does_not_execute_model():
    obs, row, suffixes, alphabet = setup()
    with pytest.raises(ValueError):
        native_batch(obs, row, suffixes, alphabet, np.ones((3, 1, 2)), layer=0)
    assert obs.calls == 0
