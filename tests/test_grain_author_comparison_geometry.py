#!/usr/bin/env python3
"""Synthetic comparisons distinguish positions, cycles and presentation."""

import numpy as np
import pytest

from grain_research.construction import math_author_comparison_geometry as M


def test_connected_registered_cycle_basis():
    e = M.graph()
    b = M.incidence(e, 24)
    c = M.cycles(e, 24)
    assert e.shape == (32, 2)
    assert c.shape == (9, 32)
    assert np.linalg.matrix_rank(c) == 9
    np.testing.assert_allclose(c @ b, 0, atol=1e-14)


def test_gradient_is_a_position_not_an_independent_validation():
    e = M.graph()
    b = M.incidence(e, 24)
    x = np.random.default_rng(226).normal(size=(24, 5))
    x -= x.mean(0)
    a = M.decompose(e, b @ x)
    np.testing.assert_allclose(a["positions"], x, atol=1e-13)
    np.testing.assert_allclose(a["remainder"], 0, atol=1e-13)
    np.testing.assert_allclose(a["cycles"], 0, atol=1e-13)


def test_non_gradient_is_retained_and_divergence_free():
    e = M.graph()
    d = np.random.default_rng(3).normal(size=(32, 5))
    a = M.decompose(e, d)
    b = M.incidence(e, 24)
    np.testing.assert_allclose(a["gradient"] + a["remainder"], d)
    np.testing.assert_allclose(b.T @ a["remainder"], 0, atol=1e-13)
    np.testing.assert_allclose(np.sum(a["gradient"] * a["remainder"], 0), 0, atol=1e-13)
    assert np.linalg.norm(a["remainder"]) > 0


def test_exchange_and_offset_both_preserved():
    d = np.arange(15).reshape(3, 5)
    bias = np.ones_like(d) * 4
    z = np.stack((d + bias, -d + bias), 1)
    odd, even = M.ordered_accounts(z)
    np.testing.assert_array_equal(odd, d)
    np.testing.assert_array_equal(even, bias)
    reverse, unchanged = M.ordered_accounts(z[:, ::-1])
    np.testing.assert_array_equal(reverse, -d)
    np.testing.assert_array_equal(unchanged, bias)


def test_no_single_reference_on_disconnected_graph():
    with pytest.raises(ValueError, match="Disconnected"):
        M.incidence(np.array([[0, 1], [2, 3]]), 4)


def test_actual_histories_not_instructions_and_no_truncation():
    a, b = [f"Left {k}" for k in range(12)], [f"Right {k}" for k in range(12)]
    text = M.render(a, b, "agreeableness")
    assert "Left 11" in text and "Right 11" in text
    assert "Writer A" in text and "Writer B" in text
    with pytest.raises(ValueError):
        M.render(a[:4], b, "agreeableness")


def test_cardinal_cycle_does_not_imply_an_ordinal_cycle():
    e = np.array([[0, 1], [0, 2], [1, 2]])
    d = np.array([[1.0], [4.0], [1.0]])
    a = M.decompose(e, d, n=3)
    assert abs(a["cycles"][0, 0]) == 2
    assert np.all(d > 0)
