#!/usr/bin/env python3
"""Synthetic, no-data checks for the marked response and availability extension."""

import numpy as np
import pytest
import torch

from grain_research.construction import math_marked_response as M
from grain_research.construction import math_recursive_collector as C


def test_binary_gate_reproduces_mask_and_zero_keys():
    generator = torch.Generator().manual_seed(120)
    q = torch.randn(2, 3, 4, 6, generator=generator)
    k, v = [torch.randn(2, 5, 2, 6, generator=generator) for _ in range(2)]
    mask = torch.tensor([[[1, 0, 1, 1, 0]] * 3] * 2, dtype=torch.bool)
    torch.testing.assert_close(
        M.gated_attention(q, k, v, mask.float()), C.gqa_attention(q, k, v, mask)
    )
    altered = v.clone()
    altered[:, [1, 4]] = 999
    torch.testing.assert_close(
        M.gated_attention(q, k, altered, mask.float()),
        M.gated_attention(q, k, v, mask.float()),
    )


def test_gate_nonempty_and_nonnegative():
    x = torch.ones(1, 1, 1, 2)
    for gate in (torch.zeros(1, 1, 1), -torch.ones(1, 1, 1)):
        with pytest.raises(ValueError):
            M.gated_attention(x, x, x, gate)


def test_availability_gradient_is_not_content_gradient():
    q = torch.zeros(1, 1, 1, 2)
    k = torch.zeros(1, 2, 1, 2)
    v = torch.tensor([[[[2.0, 0.0]], [[6.0, 0.0]]]])
    gate = torch.tensor([[[1.0, 0.5]]], requires_grad=True)
    y = M.gated_attention(q, k, v, gate)[0, 0, 0, 0]
    grad = torch.autograd.grad(y, gate)[0]
    assert grad[0, 0, 1] == pytest.approx(4 / 2.25)


def test_fixed_quadrature_polynomial_completeness_and_reverse():
    nodes, weights = M.quadrature()
    assert weights.sum() == pytest.approx(1.0)
    # Independent exact oracle F(K,V,a)=K*V*a^2, four nodes integrate its derivative.
    start, end = np.array([1.0, 2.0, 0.3]), np.array([2.0, 4.0, 1.0])
    delta, contributions = end - start, np.zeros(3)
    for t, w in zip(nodes, weights, strict=True):
        k, v, a = start + t * delta
        contributions += w * delta * np.array([v * a * a, k * a * a, 2 * k * v * a])
    actual = end[0] * end[1] * end[2] ** 2 - start[0] * start[1] * start[2] ** 2
    assert contributions.sum() == pytest.approx(actual)
    reverse = sum(
        w * (-delta) * np.array([v * a * a, k * a * a, 2 * k * v * a])
        for t, w in zip(nodes, weights, strict=True)
        for k, v, a in [end - t * delta]
    )
    np.testing.assert_allclose(reverse, -contributions)
    with pytest.raises(ValueError):
        M.quadrature(8)


def test_marked_mass_unmarked_prompt_and_connected_constant():
    a = np.array([[[[1.0, 2.0, 3.0, 7.0]]]])
    g = np.ones((1, 4, 4)) * np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([[1, 1, 1, 0]], bool)
    m = M.marked_moments(a, g, mask)
    np.testing.assert_allclose(m["mass"], 13)
    np.testing.assert_allclose(m["body_mass"], 6)
    np.testing.assert_allclose(m["untagged_mass"], 7)
    np.testing.assert_allclose(m["connected"], 0)


def test_connected_marks_depend_on_pairing_not_margins():
    a = np.array([[[[1.0, -1.0, 2.0, -2.0]]]])
    g = np.arange(16.0).reshape(1, 4, 4) ** 2
    m = M.marked_moments(a, g, np.ones((1, 4), bool))
    assert np.linalg.norm(m["connected"]) > 0
    assert not np.allclose(m["connected"], m["misaligned_connected"])
    np.testing.assert_allclose(m["constant_connected"], 0)
    shifted = M.marked_moments(a, g + 13.0, np.ones((1, 4), bool))
    np.testing.assert_allclose(m["connected"], shifted["connected"])


def test_marks_zero_and_sign():
    rng = np.random.default_rng(12)
    a, g = rng.normal(size=(2, 3, 6, 7)), rng.normal(size=(2, 7, 4))
    mask = np.ones((2, 7), bool)
    positive, negative = M.marked_moments(a, g, mask), M.marked_moments(-a, g, mask)
    for key in ("mass", "body_mass", "first", "connected", "untagged_mass"):
        np.testing.assert_allclose(negative[key], -positive[key])
    zero = M.marked_moments(a * 0, g, mask)
    assert all(np.allclose(x, 0) for x in zero.values())


def test_read_matches_explicit_frozen_numpy():
    from grain_research.construction.math_history_routing import read_current

    rng = np.random.default_rng(120)
    k, v = [rng.normal(size=(5, 2, 4)) for _ in range(2)]
    directions, queries = [rng.normal(size=(6, 4, 4)) for _ in range(2)]
    bundle = {
        "directions": directions,
        "raw_queries": queries,
        "gaps": [0] * 6,
        "frequencies": [0.0, 0.0],
        "scaling": 1.0,
    }
    expected = read_current(k, v, 0, bundle)
    actual = M.read(
        torch.tensor(k)[None, None],
        torch.tensor(v)[None, None],
        torch.tensor(queries),
        torch.tensor(directions),
    )
    np.testing.assert_allclose(actual.numpy()[0, 0], expected, atol=1e-12)
