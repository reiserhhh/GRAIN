#!/usr/bin/env python3
"""Quotient-fiber identities, rounding and local function error bound checks."""

import numpy as np
import pytest

from grain_research.construction import math_prefix_compilation as M
from scripts.diagnose_grain_math_prefix_fibers import (
    coordinate_gradient,
    reconstructed_coordinates,
    robust_pair_bound,
)


def test_triangle_bound_and_invalid_inputs():
    assert robust_pair_bound(0.2, 0, 4) == 0.1
    assert robust_pair_bound(-0.2, 0.01, 3) == pytest.approx(0.085)
    assert robust_pair_bound(0.01, 0.1, 3) == 0
    for args in ((1, -1, 1), (np.nan, 0, 1), (1, 1, -1)):
        with pytest.raises(ValueError):
            robust_pair_bound(*args)


def test_ideal_fibers_and_real_float32_coordinate_rounding():
    u = np.eye(4)[:, :2]
    h, donor = np.zeros((2, 4)), np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
    base = np.zeros(12)
    full = reconstructed_coordinates(
        base.copy(), h, donor, u, "A_property", "full_property"
    )
    projection = reconstructed_coordinates(
        base.copy(), h, donor, u, "A_property", "projection"
    )
    complement = reconstructed_coordinates(
        base.copy(), h, donor, u, "A_property", "complement"
    )
    np.testing.assert_equal(full, projection)
    np.testing.assert_equal(complement, base)
    np.testing.assert_equal(full[:2], [2, 3])


def test_compiled_gradient_exact_for_linear_and_degree_two_path():
    rng = np.random.default_rng(3)
    x, order, y = rng.normal(size=(40, 12)), np.arange(40) % 2, rng.normal(size=(40, 3))
    for arm in M.ARMS:
        phi, names = M.feature_map(x, order, arm)
        fn = M.compile_source(phi, y)
        fn["queries"], fn["columns"] = ["a", "b", "c"], names
        a, b, fixed_order = x[0], x[1], 1
        g0 = coordinate_gradient(a, fixed_order, arm, fn)
        g1 = coordinate_gradient(b, fixed_order, arm, fn)
        path = M.apply_function(
            M.feature_map(np.stack((a, b)), [fixed_order] * 2, arm)[0], fn
        )
        np.testing.assert_allclose(
            (b - a) @ (g0 + g1) / 2, path[1] - path[0], atol=1e-12
        )


def test_error_lower_bound_is_attained_by_midpoint_constant():
    assert robust_pair_bound(0.4, 0, 0) == pytest.approx(
        max(abs(0.3 - 0.1), abs(0.3 - 0.5))
    )
