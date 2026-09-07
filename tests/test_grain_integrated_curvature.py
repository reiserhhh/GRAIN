#!/usr/bin/env python3
"""Polynomial witnesses for the path-dependent orbit response formula."""

from itertools import permutations

import numpy as np
import pytest

from grain_research.vector_mechanisms.orbit_geometry import (
    integrated_hessian_contraction,
)


def test_quadratic_reduces_to_half_hessian_contraction():
    m, r = np.array([1.0, -2.0]), np.array([0.5, 3.0])
    k = np.array([[2.0, -0.3], [-0.3, -1.0]])
    result = integrated_hessian_contraction(m, r, lambda _x, v: k @ v, order=2)
    assert result == pytest.approx(0.5 * r @ k @ r)


def test_zero_local_curvature_can_have_nonzero_mean_null_response():
    m, r = np.zeros(2), np.array([1.0, -1.0])
    direction = np.array([1.0, -1.0])

    def hvp(x, v):
        return 12 * (direction @ x) ** 2 * direction * (direction @ v)

    assert np.all(hvp(m, r) == 0)
    result = integrated_hessian_contraction(m, r, hvp, order=2)
    assert result == pytest.approx(16.0)


def test_group_integral_recovers_nonquadratic_orbit_gap():
    h = np.array([-1.0, 0.4, 2.1])
    m = np.full(3, h.mean())
    weights, linear = np.array([1.0, -0.5, 0.8]), np.array([2.0, 0.3, -1.0])

    def score(x):
        return np.sum(weights * x**4) + linear @ x

    def hvp(x, v):
        return 12 * weights * x**2 * v

    states = np.array(list(permutations(h)))
    actual = np.mean([score(x) for x in states]) - score(m)
    estimated = np.mean(
        [integrated_hessian_contraction(m, x - m, hvp, order=2) for x in states]
    )
    assert estimated == pytest.approx(actual)


def test_invalid_hessian_product_is_rejected():
    with pytest.raises(ValueError, match="Hessian"):
        integrated_hessian_contraction(np.zeros(2), np.ones(2), lambda _x, _v: [np.nan])
