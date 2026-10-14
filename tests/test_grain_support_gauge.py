#!/usr/bin/env python3
"""Check masked-cache representatives as an observational equivalence class."""

import torch

from grain_research.construction import math_marked_response as M
from scripts.run_grain_support_gauge_hm120 import representative


def test_invisible_cache_representative_preserves_binary_endpoints():
    k0 = torch.tensor([[[[[0.2, 0.2]], [[8.0, 8.0]], [[0.4, 0.4]]]]])
    k1 = torch.tensor([[[[[0.5, 0.5]], [[0.6, 0.6]], [[7.0, 7.0]]]]])
    v0 = torch.tensor([[[[[1.0, 1.0]], [[9.0, 9.0]], [[2.0, 2.0]]]]])
    v1 = torch.tensor([[[[[1.0, 1.0]], [[3.0, 3.0]], [[12.0, 12.0]]]]])
    gate = [torch.tensor([[1.0, 0.0, 1.0]]), torch.tensor([[1.0, 1.0, 0.0]])]
    kk, vv = representative(k0, k1, *gate), representative(v0, v1, *gate)
    q = torch.tensor([[[[0.2, 0.4]]]])
    for side in range(2):
        original = M.gated_attention(
            q, (k0, k1)[side][:, 0], (v0, v1)[side][:, 0], gate[side][:, None]
        )
        equivalent = M.gated_attention(
            q, kk[side][:, 0], vv[side][:, 0], gate[side][:, None]
        )
        torch.testing.assert_close(original, equivalent, atol=0, rtol=0)
    midpoint = M.gated_attention(
        q, (k0 + k1)[:, 0] / 2, (v0 + v1)[:, 0] / 2, (gate[0] + gate[1])[:, None] / 2
    )
    changed = M.gated_attention(
        q,
        (kk[0] + kk[1])[:, 0] / 2,
        (vv[0] + vv[1])[:, 0] / 2,
        (gate[0] + gate[1])[:, None] / 2,
    )
    assert (midpoint - changed).abs().max() > 0.1


def test_no_mask_change_has_no_representative_change():
    x, y = torch.randn(2, 3, 4, 1, 2), torch.randn(2, 3, 4, 1, 2)
    gate = torch.tensor([[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    a, b = representative(x, y, gate, gate)
    torch.testing.assert_close(a, x, atol=0, rtol=0)
    torch.testing.assert_close(b, y, atol=0, rtol=0)
