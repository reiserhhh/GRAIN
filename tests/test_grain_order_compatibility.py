#!/usr/bin/env python3
"""Finite theoretical worlds distinguish incompatibility from fitting error."""

import numpy as np
import pytest

from grain_research.construction import math_order_compatibility as M


def test_nonlinear_monotone_null_is_compatible():
    x = np.array([-3.0, -1, 0, 0.5, 3])
    result = M.scalar_orientations(x, np.tanh(x) + 7)
    assert result["increasing"]["exact"]["bound"] == 0
    assert result["exact_either_bound"] == 0


def test_nonmonotonic_response_cannot_be_rescued_by_flipping():
    result = M.scalar_orientations(np.arange(3), [0, 2, 0])
    assert result["increasing"]["exact"]["bound"] == 1
    assert result["decreasing"]["exact"]["bound"] == 1
    assert result["exact_either_bound"] == 1


def test_inverted_scale_is_separately_reported():
    result = M.scalar_orientations(np.arange(3), [4, 2, 0])
    assert result["increasing"]["exact"]["bound"] == 2
    assert result["decreasing"]["exact"]["bound"] == 0


def test_exact_ties_and_uncertain_order_are_distinct():
    result = M.order_bound([1, 1], [0, 2], source_error=1e-9)
    assert result["exact"]["bound"] == 1
    assert result["guarded"]["bound"] is None
    near = M.order_bound([1, 1 + 1e-10], [2, 0], source_error=1e-9)
    assert near["guarded"]["constraints"] == 0


def test_empirical_adjustment_is_not_a_confidence_interval():
    result = M.order_bound([0, 1], [2, 0], reader_error=0.2)
    assert result["exact"]["bound"] == 1
    assert result["guarded"]["bound"] == 0.8


def test_partial_order_detects_domination_not_every_pair():
    x = np.array([[0, 0], [1, 1], [2, 0]])
    result = M.order_bound(x, [3, 1, 5])
    assert result["exact"]["constraints"] == 2
    assert result["guarded"]["constraints"] == 1
    assert result["guarded"]["bound"] == 1
    empty = M.order_bound([[0, 1], [1, 0]], [0, 10])
    assert empty["exact"]["bound"] is None


def test_bounds_invariant_to_observation_order_and_positive_scale():
    x, y = np.arange(5), np.array([1, 2, 0, 4, 3])
    result = M.scalar_orientations(x, y)
    perm = [4, 2, 0, 3, 1]
    changed = M.scalar_orientations(3 * x[perm] - 10, y[perm] + 100)
    assert changed["exact_either_bound"] == result["exact_either_bound"]


def test_scalar_half_gap_matches_independent_feasibility_envelope():
    rng = np.random.default_rng(249)
    for _ in range(40):
        x = rng.permutation(np.arange(10))
        y = rng.normal(size=10)
        bound = M.order_bound(x, y)["exact"]["bound"]
        order = np.argsort(x)
        lower = np.maximum.accumulate(y[order] - bound)
        assert np.max(lower - y[order] - bound) < 1e-14
        if bound > 1e-8:
            worse = np.maximum.accumulate(y[order] - bound / 2)
            assert np.max(worse - y[order] - bound / 2) > 0


def test_aliases_are_not_independent_observations():
    keys = ["a" * 64, "b" * 64, "a" * 64]
    s, y, ix = np.array([[1], [2], [1]]), np.array([[3], [4], [3]]), [0, 1, 0]
    keep, aliases = M.unique_observations(keys, s, y, ix)
    assert keep.tolist() == [0, 1]
    assert aliases == [[0, 2]]
    with pytest.raises(ValueError):
        M.unique_observations(keys, s, y, [0, 1, 2])


def test_missing_nonfinite_and_negative_discrepancies_rejected():
    for x, y in [([1], [1]), ([1, 2], [1, np.nan]), ([1, 2], [1])]:
        with pytest.raises(ValueError):
            M.order_bound(x, y)
    with pytest.raises(ValueError):
        M.order_bound([1, 2], [1, 2], source_error=-1)
