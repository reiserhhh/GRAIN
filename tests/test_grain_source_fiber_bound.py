#!/usr/bin/env python3
"""Finite sharpness and exact-equality checks, not real-data fitting."""

import numpy as np
import pytest

from scripts.bound_grain_event_behavior_hm262 import fiber_bound


def test_sharp_finite_paired_bound_and_denominator():
    """One same-cell pair attains the bound at its midpoint, the other can fit exactly."""
    x = np.array([[0] * 5, [1] * 5])
    y = np.array([[2] * 5, [7] * 5])
    floor, mask = fiber_bound(x, y)
    np.testing.assert_array_equal(mask, [True, False])
    np.testing.assert_array_equal(floor, np.full((5, 5), 0.5))
    # Midpoint errors -1/+1 on the first pair, zero on the distinct second pair.
    assert (1 + 1 + 0 + 0) / (2 * len(x)) == floor[0, 0]


def test_covector_operator_and_no_approximate_zero():
    """A small nonzero SOURCE action is not relabeled as zero to enlarge the bound."""
    x = np.array([[0] * 5, [1e-15] * 5])
    y = np.array([[1, -2, 3, -4, 5], [9] * 5])
    floor, mask = fiber_bound(x, y)
    np.testing.assert_array_equal(mask, [True, False])
    a = np.array([2, 1, 0, -1, 3])
    assert a @ floor @ a == (a @ y[0]) ** 2 / (4 * len(x))
    with pytest.raises(ValueError, match="Missing"):
        fiber_bound(x, y * np.nan)
