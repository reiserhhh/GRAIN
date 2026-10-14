#!/usr/bin/env python3
"""Check row blocking preserves all keys, causal mask, padding and GQA."""

from types import SimpleNamespace

import pytest
import torch

from grain_research.construction.math_full_attention_rows import (
    BACKEND,
    full_attention_rows,
    register,
)


@pytest.mark.parametrize("groups,rows", [(1, 1), (2, 3), (3, 20)])
def test_full_key_eager_equivalence(groups, rows):
    """Block boundaries never change the key softmax denominator."""
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

    torch.manual_seed(227)
    q = torch.randn(2, groups * 2, 11, 4)
    k, v = torch.randn(2, 2, 11, 4), torch.randn(2, 2, 11, 4)
    mask = torch.triu(torch.full((2, 1, 11, 11), torch.finfo(q.dtype).min), 1)
    mask[1, :, :, 8:] = torch.finfo(q.dtype).min
    module = SimpleNamespace(
        config=SimpleNamespace(grain_query_rows=rows),
        training=False,
        num_key_value_groups=groups,
    )
    expected, _ = eager_attention_forward(module, q, k, v, mask, scaling=0.5)
    actual, weights = full_attention_rows(module, q, k, v, mask, scaling=0.5)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert weights is None
    v2 = v.clone()
    v2[1, :, 8:] += 100
    actual2, _ = full_attention_rows(module, q, k, v2, mask, scaling=0.5)
    torch.testing.assert_close(actual2[1, :8], actual[1, :8])


def test_registry_keeps_original_causal_mask():
    """A custom HF backend must not silently discard the original mask."""
    from transformers import Qwen2Config
    from transformers.masking_utils import create_causal_mask

    cfg = Qwen2Config()
    embeds = torch.zeros(2, 7, 4)
    padding = torch.tensor([[1] * 7, [1] * 4 + [0] * 3])
    cfg._attn_implementation = "eager"
    old = create_causal_mask(cfg, embeds, padding, past_key_values=None)
    register()
    cfg._attn_implementation = BACKEND
    new = create_causal_mask(cfg, embeds, padding, past_key_values=None)
    assert torch.equal(new, old)
    assert new[0, 0, 1, 2] < -1e20
    assert new[1, 0, 6, 4] < -1e20


@pytest.mark.parametrize("bad", ["none", "boolean", "dropout", "cached"])
def test_unsupported_observation_rejected(bad):
    """Unsafe mask/caching changes cannot masquerade as execution equivalence."""
    module = SimpleNamespace(
        config=SimpleNamespace(grain_query_rows=3),
        training=False,
        num_key_value_groups=1,
    )
    q = torch.zeros(1, 2, 5, 3)
    k, v = q.clone(), q.clone()
    mask = torch.zeros(1, 1, 5, 5)
    dropout = 0.0
    if bad == "none":
        mask = None
    elif bad == "boolean":
        mask = mask.bool()
    elif bad == "dropout":
        dropout = 0.1
    else:
        k, v = k[:, :, :3], v[:, :, :3]
    with pytest.raises(ValueError):
        full_attention_rows(module, q, k, v, mask, scaling=1.0, dropout=dropout)
