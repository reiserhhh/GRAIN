#!/usr/bin/env python3
"""The descent bound is a mathematical oracle, not model selection."""

import numpy as np
import pytest

from scripts.run_grain_math_fiber_descent_bound import bound


def test_exact_conditional_variance():
    x = np.array([[1.0, 3.0], [-2.0, 2.0]])
    assert bound(x)["h_only_mse_lower_bound"] == pytest.approx(2.5)


def test_every_common_predictor_respects_bound():
    x = np.random.default_rng(12).normal(size=(2, 2, 5))
    mu = x.mean(axis=1)
    for error in (0.0, 0.1, 4.0):
        prediction = mu + error
        loss = np.sum((x - prediction[:, None]) ** 2) / 4
        assert loss + 1e-14 >= bound(x)["h_only_mse_lower_bound"]


def test_equal_action_and_context_level_offset():
    assert bound(np.array([[2.0, 2.0], [5.0, 5.0]]))["h_only_mse_lower_bound"] == 0


def test_no_information_is_not_a_fabricated_ratio():
    assert bound(np.zeros((2, 2)))["bound_relative_to_observed_rms"] is None


def test_receiver_swap_and_units():
    x = np.array([[1.0, 3.0], [4.0, 8.0]])
    assert (
        bound(x[:, ::-1])["h_only_mse_lower_bound"]
        == bound(x)["h_only_mse_lower_bound"]
    )
    assert (
        bound(2 * x)["h_only_mse_lower_bound"] == 4 * bound(x)["h_only_mse_lower_bound"]
    )
