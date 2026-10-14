#!/usr/bin/env python3
"""Synthetic tests distinguish context interaction from output-link curvature."""

import numpy as np
import pytest

from grain_research.construction.math_context_response import (
    context_characters,
    context_pair_account,
    sigmoid,
)


def test_four_corners_reconstruct():
    y = np.arange(12.0).reshape(3, 2, 2) ** 2
    m, b, c, i = context_characters(y)
    for body, bs in enumerate((-1, 1)):
        for context, cs in enumerate((-1, 1)):
            np.testing.assert_allclose(
                y[:, body, context], m + bs * b + cs * c + bs * cs * i
            )


def test_pure_offset_is_not_a_body_context_interaction():
    old = np.array([-1.0, 2.0])
    account = context_pair_account(old, old + 2)
    assert account["joint_character"] == 0
    assert account["log_odds_gap_change"] == 0
    assert abs(account["probability_gap_change"]) > 0.1
    assert account["joint_probability_remainder"] == pytest.approx(0)


def test_body_context_can_reverse_pair_gap():
    account = context_pair_account([-1, 1], [1, -1])
    assert account["joint_character"] == -1
    assert account["log_odds_gap_change"] == -4
    assert account["old_probability_gap"] > 0
    assert account["common_probability_gap"] < 0


def test_floor_is_not_a_fitted_model_error():
    old, common = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    mean = (sigmoid(old) + sigmoid(common)) / 2
    risk = np.mean((sigmoid(old) - mean) ** 2 + (sigmoid(common) - mean) ** 2) / 2
    assert context_pair_account(old, common)[
        "body_only_two_context_floor"
    ] == pytest.approx(risk)


def test_missing_corners_are_not_zero_filled():
    with pytest.raises(ValueError):
        context_characters([[0, 1], [2, np.nan]])
    with pytest.raises(ValueError):
        context_pair_account([1], [2])


def test_sigmoid_preserves_extreme_finite_inputs():
    np.testing.assert_allclose(sigmoid([-1000, 0, 1000]), [0, 0.5, 1])
