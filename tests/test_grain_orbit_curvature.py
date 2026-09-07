#!/usr/bin/env python3
"""Finite-difference accuracy and fixed mean-null radial construction."""

import numpy as np
import torch

from scripts.run_grain_orbit_curvature import coefficients, radial_state
from scripts.run_grain_orbit_response import apply_permutation


def test_central_coefficients_recover_quadratic():
    values = {x: 0.4 + 2 * x - 3 * x * x for x in (0.0, -0.125, 0.125)}
    assert np.allclose(coefficients(values, 0.125), (2.0, -3.0))


def test_richardson_cancels_cubic_and_quartic_coefficient_bias():
    f = lambda x: 0.4 + 2 * x - 3 * x * x + 7 * x**3 + 11 * x**4
    values = {x: f(x) for x in (0.0, -0.125, 0.125, -0.0625, 0.0625)}
    big, small = coefficients(values, 0.125), coefficients(values, 0.0625)
    result = [(4 * a - b) / 3 for a, b in zip(small, big, strict=True)]
    assert np.allclose(result, [2.0, -3.0])


def test_radial_path_preserves_comment_means_at_negative_and_positive_radii():
    h = torch.arange(40, dtype=torch.float64).reshape(10, 4)
    groups = [[0, 1, 2, 3], [5, 6, 7]]
    for a in (-1.0, -0.5, 0.0, 0.125, 1.0):
        x = radial_state(h, groups, [0, 1], 731, a)
        for g in groups:
            assert torch.allclose(x[g].mean(0), h[g].mean(0))
        assert torch.equal(x[[4, 8, 9]], h[[4, 8, 9]])
    assert torch.allclose(
        radial_state(h, groups, [0, 1], 731, 1.0),
        apply_permutation(h, groups, [0, 1], 731),
    )


def test_even_odd_split_is_exact_without_assuming_quadratic():
    f = lambda x: np.sin(x) + np.exp(x * x)
    x = 0.5
    even, odd = (f(x) + f(-x)) / 2 - f(0), (f(x) - f(-x)) / 2
    assert np.isclose(even + odd, f(x) - f(0))
