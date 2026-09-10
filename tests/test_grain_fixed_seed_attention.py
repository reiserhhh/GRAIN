#!/usr/bin/env python3
"""Compare the analytic rotary/GQA channels with independent torch autograd."""

import numpy as np
import pytest
import torch

from grain_research.vector_mechanisms.fixed_seed_attention import (
    decompose_attention,
    rotary,
)


@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("seed", [0, 1])
def test_rotary_grouped_channels_recover_nonquery_body_gradient(kv_heads, seed):
    torch.manual_seed(seed)
    n, width, heads, d = 7, 6, 4, 4
    h = torch.randn(n, width, dtype=torch.float64, requires_grad=True)
    wq = torch.randn(heads * d, width, dtype=torch.float64) / 3
    wk = torch.randn(kv_heads * d, width, dtype=torch.float64) / 3
    wv = torch.randn_like(wk)
    q = (h @ wq.T + 0.3).reshape(n, heads, d)
    k = (h @ wk.T - 0.2).reshape(n, kv_heads, d)
    v = (h @ wv.T + 0.1).reshape(n, kv_heads, d)
    theta = torch.randn(n, d // 2, dtype=torch.float64).repeat(1, 2)
    cos, sin = theta.cos(), theta.sin()

    def rot(x):
        a, b = x.chunk(2, dim=-1)
        return x * cos[:, None] + torch.cat([-b, a], -1) * sin[:, None]

    qr, kr = rot(q), rot(k).repeat_interleave(heads // kv_heads, dim=1)
    vr = v.repeat_interleave(heads // kv_heads, dim=1)
    mass = torch.einsum("hd,thd->ht", qr[-1], kr).div(d**0.5).softmax(-1)
    output = torch.einsum("ht,thd->hd", mass, vr)
    g = torch.randn(heads, d, dtype=torch.float64)
    grad = torch.autograd.grad((output * g).sum(), h)[0].detach().numpy()
    arrays = [x.detach().numpy() for x in (q[-1], k, v, g, cos, sin, wk, wv)]
    result = decompose_attention(*arrays)
    np.testing.assert_allclose(
        result["output"], output.detach().numpy().ravel(), atol=1e-12
    )
    combined = result["key_field"] + result["value_field"]
    np.testing.assert_allclose(combined[:-1], grad[:-1], atol=1e-12)
    assert np.linalg.norm(combined[-1] - grad[-1]) > 1e-5


def test_rotary_transpose_is_adjoint_even_with_unpaired_coefficients():
    rng = np.random.default_rng(39)
    x, y, c, s = (rng.normal(size=(5, 4)) for _ in range(4))
    assert np.sum(rotary(x, c, s) * y) == pytest.approx(
        np.sum(x * rotary(y, c, s, transpose=True)), abs=1e-12
    )
