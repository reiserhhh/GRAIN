#!/usr/bin/env python3
"""Finite injectivity controls distinguish raw vector means and reduced means."""

import numpy as np
import pytest

from grain_research.construction import math_author_moment_measurement as M
from scripts.audit_grain_math_mean_embedding_moments import analyze_map


def test_full_map_recovers_higher_order_distribution():
    _, masks, w = M.cube(3)
    c = np.vstack([np.zeros(7), np.eye(7)])
    r = analyze_map(c)
    pp, pm = M.parity_profiles(masks[-1], dimension=3)
    np.testing.assert_allclose(M.moment_design(2, 3) @ pp, M.moment_design(2, 3) @ pm)
    assert r["full_rank"]
    for p in (pp, pm):
        recovered = (p @ (w @ c) - c[0]) @ r["right_inverse"]
        np.testing.assert_allclose(recovered, M.moments(p, 3)[1:])


def test_first_order_projection_loses_higher_moments():
    _, masks, w = M.cube(3)
    c = np.zeros((8, 3))
    c[masks.sum(1) == 1] = np.eye(3)
    r = analyze_map(c)
    pp, pm = M.parity_profiles(masks[-1], dimension=3)
    assert not r["full_rank"]
    assert r["rank"] == 3
    np.testing.assert_allclose(pp @ (w @ c), pm @ (w @ c))


def test_weak_but_nonzero_direction_is_retained_without_ridge():
    c = np.vstack([np.zeros(3), np.diag([1, 0.1, 0.0001])])
    r = analyze_map(c)
    assert r["full_rank"]
    assert r["inverse_norm"] == pytest.approx(10000)


def test_invalid_map_fails():
    with pytest.raises(ValueError):
        analyze_map(np.array([[np.nan], [1]]))


def test_constant_map_has_no_identifiable_distribution_direction():
    r = analyze_map(np.zeros((8, 3)))
    assert r["rank"] == 0
    assert not r["full_rank"]
    assert r["retained_condition"] is None
    assert r["inverse_norm"] is None
    np.testing.assert_array_equal(r["right_inverse"], np.zeros((3, 7)))
