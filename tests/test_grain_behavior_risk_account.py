#!/usr/bin/env python3
"""Synthetic exact-risk checks,not acceptance through outcome selection."""

import numpy as np

from scripts.describe_grain_behavior_pushforward_hm291 import account


def test_risk_identity_under_reference_shift():
    """Source/application mean shifts remain explicit instead of centering away."""
    y = np.array([[4.0, 9.0], [1.0, 3.0], [8.0, -2.0]])
    p = y * 2 + 4
    parts, gain, _stranger, error = account(
        y, p, np.array([1.0, 2.0]), np.array([3.0, 5.0]), np.array([2, 0, 1])
    )
    np.testing.assert_allclose(parts.sum(0), gain)
    assert error < 1e-12


def test_own_stranger_is_alignment_not_spread():
    """A bijection cancels all marginal prediction-amplitude terms."""
    y = np.array([[0.0], [1.0], [2.0]])
    p = 10 * y
    _, gain, stranger, error = account(
        y, p, np.zeros(1), np.ones(1), np.array([2, 0, 1])
    )
    assert gain[0] < 0 < stranger[0]
    assert error < 1e-12
