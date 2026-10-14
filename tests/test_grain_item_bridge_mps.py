#!/usr/bin/env python3
"""Confirm that CPU64 normalization changes neither native states nor readings."""

import numpy as np
import pytest

from grain_research.construction import math_item_bridge as M
from grain_research.construction import math_item_bridge_mps as C


def test_companion_differs_only_in_normalizer_compute_location():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(255)
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    ids = [[1, 2, 3, 8], [1, 2, 3, 10, 11], [1, 2, 3, 9, 7]]
    a, b = (
        M.native_items(model, ids, [0, 1], "cpu", 2),
        C.native_items(model, ids, [0, 1], "cpu", 2),
    )
    for name in ("states", "logits", "top", "prefix_tokens"):
        assert np.array_equal(a[name], b[name])
    assert np.allclose(a["normalizer"], b["normalizer"], rtol=0, atol=1e-14)
