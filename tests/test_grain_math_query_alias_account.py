#!/usr/bin/env python3
"""Exact aliasing can amplify or conceal distinct conditional actions."""

import numpy as np
import pytest

from grain_research.construction import math_query_cube as U
from scripts.run_grain_math_query_alias_account import account


@pytest.mark.parametrize("sign", [-1, 1])
def test_r_and_q_can_cancel_or_amplify_on_natural_diagonal(sign):
    x = (U.characters()[2] + sign * U.characters()[1]).reshape(2, 2, 2)
    diagonal, _, stats = account(x, U.split(x)[0])
    assert stats["full_cube_H_only_mse_bound"] == 2
    assert stats["R_Q_signed_cross"] == 2 * sign
    assert stats["diagonal_H_only_mse_bound"] == (0 if sign == -1 else 4)
    if sign == -1:
        np.testing.assert_array_equal(diagonal, np.zeros((2, 2)))


def test_higher_interactions_alias_to_diagonal_mean_and_h():
    x = (U.characters()[3] + 2 * U.characters()[7]).reshape(2, 2, 2)
    _, collapsed, stats = account(x, U.split(x)[0])
    np.testing.assert_array_equal(collapsed, [1, 0, 2, 0])
    assert stats["full_cube_H_only_mse_bound"] == 5
    assert stats["diagonal_H_only_mse_bound"] == 0


def test_random_vector_alias_and_conditional_variance_identities():
    x = np.random.default_rng(33).normal(size=(2, 2, 2, 4, 6))
    _, _, stats = account(x, U.split(x)[0])
    for name, value in stats.items():
        if name.endswith("_error"):
            assert value < 1e-13
