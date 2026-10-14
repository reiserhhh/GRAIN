#!/usr/bin/env python3
"""Failure-capable observation-law controls, not personality accuracy tests."""

import numpy as np
import pytest

from grain_research.construction import math_author_contrast as M


def test_condition_choice_and_response_are_distinct():
    c, x = np.array([-1, -1, 1, 1]), np.array([-1, 1, -1, 1])
    p = np.full(4, 0.25)
    a = M.condition_account(c + c * x, p, c)
    np.testing.assert_allclose(a["choice"], c)
    assert a["total_norm2"] == pytest.approx(2)
    assert a["choice_norm2"] == pytest.approx(1)
    assert a["within_norm2"] == pytest.approx(1)
    assert a["cross"] == pytest.approx(0)


def test_shared_channel_loses_specific_contrast():
    f, p = np.array([1, -1, -1, 1]), np.full(4, 0.25)
    k = np.eye(2)[[0, 1, 0, 1]]
    out = M.channel_account(f, p, k)
    assert out["visible_norm2"] == pytest.approx(0)
    assert out["lost_norm2"] == pytest.approx(1)
    assert out["closure_error"] < 1e-14


def test_invertible_reading_address_preserves_contrast():
    p, f = np.array([0.1, 0.2, 0.3, 0.4]), np.array([2, -1, 0, 0])
    k = np.eye(4)[[3, 1, 0, 2]]
    out = M.channel_account(f, p, k)
    assert out["visible_norm2"] == pytest.approx(M.norm2(f, p))
    assert out["lost_norm2"] == pytest.approx(0)


def test_stochastic_channel_not_an_orthogonal_projection_on_input():
    p, f = np.full(2, 0.5), np.array([-1, 1])
    out = M.channel_account(f, p, [[0.8, 0.2], [0.2, 0.8]])
    assert out["visible_norm2"] == pytest.approx(0.36)
    assert out["lost_norm2"] == pytest.approx(0.64)


def test_source_target_closure_and_invisible_condition():
    c, x = np.array([-1, -1, 1, 1]), np.array([-1, 1, -1, 1])
    p = np.full(4, 0.25)
    targets = np.column_stack((x, c * x))
    proj = M.source_projection(x[:, None], p)
    out = M.target_account(c * x, targets, p, proj)
    assert out["visible_norm2"] < 1e-28
    np.testing.assert_allclose(out["truth"], [0, 1])
    assert out["target_closure_error"] < 1e-14
    proj2 = M.source_projection(np.column_stack((x, c, c * x)), p)
    assert M.target_account(c * x, targets, p, proj2)["lost_norm2"] < 1e-28


def test_source_coordinate_change_does_not_change_span():
    rng = np.random.default_rng(82)
    b, f, p = rng.normal(size=(20, 3)), rng.normal(size=20), np.full(20, 0.05)
    t = np.array([[2, 0, 1], [0, 3, 1], [0, 0, 4]])
    np.testing.assert_allclose(
        M.project(f, p, M.source_projection(b, p)),
        M.project(f, p, M.source_projection(b @ t + 7, p)),
        atol=1e-12,
    )


def test_nonlinear_recoding_before_mean_is_not_span_invariance():
    x, p = np.array([-2, -1, 1, 2.0]), np.full(4, 0.25)
    original = M.project(x, p, M.source_projection(x[:, None], p))
    recoded = M.project(x, p, M.source_projection(x[:, None] ** 3, p))
    assert np.max(abs(original - recoded)) > 0.1


def test_zero_columns_and_singular_dictionary_are_not_rescued():
    p = np.full(4, 0.25)
    b = np.column_stack((np.ones(4), [-1, 1, -1, 1], [-2, 2, -2, 2]))
    assert M.source_projection(b, p)["rank"] == 1


def test_cold_empty_and_outside_support():
    values = np.arange(12).reshape(4, 3)
    assert M.cold_score(values, np.array([], dtype=int))["scores"] is None
    with pytest.raises(ValueError, match="outside"):
        M.cold_score(values, [4])
    with pytest.raises(ValueError, match="integer"):
        M.cold_score(values, [0.2])
    np.testing.assert_allclose(M.cold_score(values, [0, 1])["scores"], [1.5, 2.5, 3.5])


def test_invalid_reference_and_channel_fail():
    with pytest.raises(ValueError, match="positive"):
        M.probability([0, 1])
    with pytest.raises(ValueError, match="sum"):
        M.probability([0.2, 0.2])
    with pytest.raises(ValueError, match="stochastic"):
        M.channel_account([1, -1], [0.5, 0.5], [[1, 1], [0, 1]])
