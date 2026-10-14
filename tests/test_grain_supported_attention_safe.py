#!/usr/bin/env python3
"""Stable attention checks include the actual zero-mass overflow failure class."""

import numpy as np
import torch

from grain_research.construction import math_marked_response as M
from grain_research.construction import math_supported_attention_safe as S
from tests.test_grain_supported_measure import fixture


def test_zero_mass_large_logit_is_not_zero_times_infinity():
    q = torch.tensor([[[[100.0, 100.0]]]])
    k = torch.tensor([[[[0.0, 0.0]], [[100.0, 100.0]]]])
    v = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    mass = torch.tensor([[[1.0, 0.0]]])
    assert not torch.isfinite(M.gated_attention(q, k, v, mass)).all()
    actual = S.attention(q, k, v, mass)
    torch.testing.assert_close(actual, v[:, :1], atol=0, rtol=0)


def test_safe_positive_mass_reading_and_derivative_match_old_when_finite():
    gen = torch.Generator().manual_seed(1212)
    q, k, v = [
        torch.randn(1, 2, 1, 2, generator=gen, dtype=torch.float64) for _ in range(3)
    ]
    mass = torch.tensor(
        [[[0.2, 0.8], [0.3, 0.7]]], dtype=torch.float64, requires_grad=True
    )
    old, new = M.gated_attention(q, k, v, mass), S.attention(q, k, v, mass)
    go = torch.autograd.grad(old.square().sum(), mass)[0]
    gn = torch.autograd.grad(new.square().sum(), mass)[0]
    torch.testing.assert_close(old, new, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(go, gn, atol=1e-14, rtol=1e-14)


def test_safe_measure_matches_finite_original_and_keeps_rotary_origin():
    k0, k1, v0, v1, a, b, *extra = fixture()
    k, v, origin, delta = S.pack(k0, k1, v0, v1, a, b)
    for t, hk, hv, mass in ((0, k0, v0, a), (1, k1, v1, b)):
        new = S.construct(k, v, origin + t * delta, np.tile(np.arange(3), 2), 3, *extra)
        old = M.construct(hk, hv, mass, *extra)
        for x, y in zip(new, old, strict=True):
            torch.testing.assert_close(x, y, atol=3e-7, rtol=2e-6)
