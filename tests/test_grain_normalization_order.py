#!/usr/bin/env python3
"""Finite math witnesses for conversion of within-group dispersion into mean shifts."""

import numpy as np
import torch

from grain_research.vector_mechanisms.normalization_order import (
    group_projection,
    normalization_order,
    quadratic_order,
    rms_normalize,
)


def test_order_difference_is_group_constant_and_zero_outside():
    rng = np.random.default_rng(2701)
    x = rng.normal(size=(9, 6))
    groups, gain = [[0, 1, 2], [5, 6]], rng.uniform(0.5, 2, 6)
    shift = normalization_order(x, groups, gain, 0.1)
    np.testing.assert_allclose(group_projection(shift, groups), shift, atol=1e-14)
    np.testing.assert_allclose(shift[[3, 4, 7, 8]], 0, atol=1e-14)
    assert np.linalg.norm(shift) > 0.1
    constant = group_projection(x, groups)
    np.testing.assert_allclose(
        normalization_order(constant, groups, gain, 0.1), 0, atol=1e-14
    )


def test_quadratic_formula_equals_autograd_hessian_contraction():
    rng = np.random.default_rng(2702)
    mean, gain = rng.normal(size=4), rng.uniform(0.5, 2, 4)
    matrix = rng.normal(size=(4, 4))
    covariance = matrix @ matrix.T
    mu, gamma = torch.tensor(mean), torch.tensor(gain)
    expected = []
    for j in range(4):

        def f(x, j=j):
            return (gamma * x / torch.sqrt(torch.mean(x**2) + 0.2))[j]

        hessian = torch.autograd.functional.hessian(f, mu).numpy()
        expected.append(-0.5 * np.sum(hessian * covariance))
    np.testing.assert_allclose(
        quadratic_order(mean, covariance, gain, 0.2), expected, rtol=1e-12, atol=1e-12
    )


def test_small_symmetric_dispersion_has_fourth_order_truncation_error():
    rng = np.random.default_rng(2703)
    mean, gain = rng.normal(size=8), np.ones(8)
    directions = rng.normal(size=(30, 8))
    directions = np.vstack([directions, -directions])
    errors = []
    for scale in (0.04, 0.02, 0.01):
        delta = scale * directions
        exact = rms_normalize(mean, gain, 0.2) - rms_normalize(
            mean + delta, gain, 0.2
        ).mean(0)
        approx = quadratic_order(mean, delta.T @ delta / len(delta), gain, 0.2)
        errors.append(np.linalg.norm(exact - approx))
    assert errors[0] / errors[1] > 12
    assert errors[1] / errors[2] > 12
