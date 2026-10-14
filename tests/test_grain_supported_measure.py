#!/usr/bin/env python3
"""Result-independent support/position/mass-action identities on a tiny decoder."""

import numpy as np
import pytest
import torch

from grain_research.construction import math_marked_response as M
from grain_research.construction import math_supported_measure as S
from tests.test_grain_marked_response_constructor import tiny_case


def fixture():
    """Two distinct endpoint histories sharing immutable toy pretrained maps."""
    k, v, seeds, cos, sin, weights, config = tiny_case()
    a = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    b = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    return k, k + 0.3, v, v - 0.4, a, b, seeds, cos, sin, weights, config


def run(k, v, mass, positions, extra):
    """Apply a positioned atom constructor with the original current origin."""
    seeds, cos, sin, weights, config = extra
    return S.construct(k, v, mass, positions, 3, seeds, cos, sin, weights, config)


def test_measure_endpoints_match_binary_original_without_rotary_shift():
    k0, k1, v0, v1, a, b, *extra = fixture()
    k, v, origin, delta = S.pack(k0, k1, v0, v1, a, b)
    for t, oldk, oldv, olda in ((0, k0, v0, a), (1, k1, v1, b)):
        actual = run(k, v, origin + t * delta, np.tile(np.arange(3), 2), extra)
        expected = M.construct(oldk, oldv, olda, *extra)
        for x, y in zip(actual, expected, strict=True):
            torch.testing.assert_close(x, y, atol=3e-7, rtol=2e-6)


def test_zero_support_representatives_do_not_enter_any_mixture_point():
    k0, k1, v0, v1, a, b, *extra = fixture()
    first = S.pack(k0, k1, v0, v1, a, b)
    altered = [
        x + (mass[:, None, :, None, None] == 0) * 1e8
        for x, mass in ((k0, a), (k1, b), (v0, a), (v1, b))
    ]
    second = S.pack(*altered, a, b)
    for x, y in zip(first, second, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    for t in (0.0, 0.25, 0.5, 1.0):
        actual = [
            run(k, v, o + t * d, np.tile(np.arange(3), 2), extra)
            for k, v, o, d in (first, second)
        ]
        for x, y in zip(*actual, strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_atom_refinement_preserves_positioned_recursive_observation():
    k0, k1, v0, v1, a, b, *extra = fixture()
    k, v, origin, delta = S.pack(k0, k1, v0, v1, a, b)
    mass = origin + 0.4 * delta
    positions = np.tile(np.arange(3), 2)
    first = run(k, v, mass, positions, extra)
    second = run(
        k.repeat_interleave(2, 2),
        v.repeat_interleave(2, 2),
        mass.repeat_interleave(2, 1) / 2,
        np.repeat(positions, 2),
        extra,
    )
    for x, y in zip(first, second, strict=True):
        torch.testing.assert_close(x, y, atol=3e-7, rtol=2e-6)


def test_atom_reordering_moves_positions_and_mass_with_atoms():
    k0, k1, v0, v1, a, b, *extra = fixture()
    k, v, origin, delta = S.pack(k0, k1, v0, v1, a, b)
    mass = origin + 0.4 * delta
    positions, ix = np.tile(np.arange(3), 2), np.array([4, 1, 5, 2, 0, 3])
    first = run(k, v, mass, positions, extra)
    second = run(k[:, :, ix], v[:, :, ix], mass[:, ix], positions[ix], extra)
    for x, y in zip(first, second, strict=True):
        torch.testing.assert_close(x, y, atol=3e-7, rtol=2e-6)


def test_current_origin_not_atom_count_and_bad_positions_are_rejected():
    k0, k1, v0, v1, a, b, *extra = fixture()
    k, v, origin, _ = S.pack(k0, k1, v0, v1, a, b)
    with pytest.raises(ValueError, match="old positions"):
        run(k, v, origin, np.arange(6), extra)


def test_negative_original_mass_is_rejected():
    k0, k1, v0, v1, a, b, *_ = fixture()
    with pytest.raises(ValueError, match="nonnegative"):
        S.pack(k0, k1, v0, v1, -a, b)


def test_mass_derivative_has_centered_value_formula():
    q = torch.tensor([[[[0.2, 0.4]]]], dtype=torch.float64)
    k = torch.tensor([[[[0.3, -0.2]], [[0.1, 0.7]]]], dtype=torch.float64)
    v = torch.tensor([[[[1.0, 2.0]], [[-1.0, 3.0]]]], dtype=torch.float64)
    mass = torch.tensor([[[0.4, 0.8]]], dtype=torch.float64, requires_grad=True)
    out = M.gated_attention(q, k, v, mass)
    actual = torch.autograd.grad(out.sum(), mass)[0].flatten()
    potential = torch.exp(
        torch.einsum("bd,id->bi", q[0, :, 0], k[0, :, 0]) / 2**0.5
    ).flatten()
    expected = (
        potential
        * (v[0, :, 0] - out[0, 0, 0]).sum(-1)
        / (potential * mass.flatten()).sum()
    )
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)


def test_mass_only_recursive_gradient_is_not_detached():
    k0, k1, v0, v1, a, b, *extra = fixture()
    k, v, origin, delta = S.pack(k0, k1, v0, v1, a, b)
    mass = (origin + 0.5 * delta).requires_grad_(True)
    value = sum(
        x.square().sum() for x in run(k, v, mass, np.tile(np.arange(3), 2), extra)
    )
    derivative = (torch.autograd.grad(value, mass)[0] * delta).sum()
    assert torch.isfinite(derivative) and abs(derivative) > 0
    step = 0.02
    ends = [
        sum(
            x.square().sum()
            for x in run(k, v, origin + t * delta, np.tile(np.arange(3), 2), extra)
        )
        for t in (0.5 - step, 0.5 + step)
    ]
    torch.testing.assert_close(
        derivative, (ends[1] - ends[0]) / (2 * step), atol=2e-5, rtol=2e-3
    )


def test_mark_moments_obey_affine_change_and_constant_control():
    rng = np.random.default_rng(121)
    action, marks = rng.normal(size=(2, 2, 6, 8)), rng.normal(size=(2, 8, 4))
    visible, sides = np.ones((2, 8), bool), np.repeat([0, 1], 4)
    first = S.moments(action, marks, visible, sides)
    L, offset = rng.normal(size=(4, 4)), rng.normal(size=4)
    second = S.moments(action, marks @ L.T + offset, visible, sides)
    np.testing.assert_allclose(
        second["first"],
        first["first"] @ L.T + first["mass"][..., None] * offset,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        second["connected"], first["connected"] @ L.T, atol=1e-13
    )
    constant = S.moments(action, np.ones_like(marks), visible, sides)
    np.testing.assert_allclose(constant["connected"], 0, atol=1e-13)
