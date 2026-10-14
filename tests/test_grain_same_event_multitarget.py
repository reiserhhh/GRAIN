#!/usr/bin/env python3
"""Finite observation tests, not personality validity assertions."""

import numpy as np
import pytest
from scipy.special import softmax

from grain_research.construction import math_same_event_multitarget as M


def test_digit_units_and_gauge():
    a = np.arange(20).reshape(2, 10) / 5
    z = np.array([9.0, 10.0])
    u = M.digit_units(a, z)
    np.testing.assert_allclose(u[:, 0], softmax(a, axis=1) @ (np.arange(10) / 9))
    np.testing.assert_allclose(M.digit_units(a + 3, z + 3), u)


def test_shared_response_has_zero_specific_component():
    x = np.repeat(np.arange(6).reshape(2, 1, 3), 5, axis=1)
    common, specific = M.target_account(x)
    np.testing.assert_array_equal(common, x)
    np.testing.assert_array_equal(specific, np.zeros_like(x))


def test_specific_and_head_account():
    rng = np.random.default_rng(199)
    h, w = rng.normal(size=(4, 5, 8)), rng.normal(size=8)
    common, specific = M.target_account(h)
    np.testing.assert_allclose(h @ w, common @ w + specific @ w, atol=2e-14)
    np.testing.assert_allclose(specific.mean(1), 0, atol=2e-14)


def test_covariance_addition_not_correlation_addition():
    rng = np.random.default_rng(198)
    x, a, b = (rng.normal(size=(24, 5)) for _ in range(3))
    np.testing.assert_allclose(
        M.covariances(x, a + b), M.covariances(x, a) + M.covariances(x, b)
    )
    assert not np.allclose(
        M.correlations(x, a + b), M.correlations(x, a) + M.correlations(x, b)
    )


def test_perfect_and_undefined_contacts():
    x = np.column_stack((np.arange(8), np.ones(8)))
    r = M.correlations(x, x)
    assert r[0, 0] == pytest.approx(1) and np.isnan(r[1, 1])


def test_natural_vector_identity():
    x = np.arange(36).reshape(12, 3)
    s, c = np.arange(4), np.arange(4, 8)
    d = M.vector_actions(x, s, c)
    np.testing.assert_array_equal(d[1], -2 * d[0])
    np.testing.assert_allclose(d[0], x[4:].mean(0) - x[np.r_[0:4, 8:12]].mean(0))
    with pytest.raises(ValueError, match="disjoint"):
        M.vector_actions(x, s, s)


def test_arm_actions_are_complete():
    x = np.arange(15).reshape(5, 3)
    a = M.natural_actions(x)
    np.testing.assert_array_equal(a, [x[1] - x[2], x[3] - x[4]])
    with pytest.raises(ValueError, match="Five complete"):
        M.natural_actions(x[:4])


def test_cluster_bootstrap_is_reproducible():
    x = np.arange(120).reshape(24, 5)
    np.testing.assert_array_equal(
        M.bootstrap_diagonal(x, x, 31, 199), M.bootstrap_diagonal(x, x, 31, 199)
    )
