#!/usr/bin/env python3
"""Paired-query identities and an integrated cross-Hessian finite witness."""

import numpy as np
import pytest

from scripts.report_grain_cross_token_reading import paired_account


def test_equal_author_account_is_not_token_weighted():
    a = [np.ones(2), 2 * np.ones(20)]
    b = [np.ones(2), -2 * np.ones(20)]
    s = paired_account(a, b)
    assert s["left_energy"] == pytest.approx(2.5)
    assert s["common_energy"] == pytest.approx(0.5)
    assert s["difference_fraction"] == pytest.approx(0.8)
    assert s["author_cosine_mean"] == pytest.approx(0)
    assert s["energy_identity_error"] < 1e-12
    np.testing.assert_allclose(paired_account(b, a)["difference_fraction"], 0.8)


def test_finite_reading_change_is_integrated_mixed_derivative():
    """f(a,b)=a^3 b^2: finite change equals a path integral, not the point Hessian."""
    a, b, epsilon = 0.7, -0.4, 0.15
    gradient_b = lambda x: 2 * b * x**3
    finite = (gradient_b(a + epsilon) - gradient_b(a - epsilon)) / 2
    nodes, weights = np.polynomial.legendre.leggauss(3)
    integrated = epsilon / 2 * np.sum(weights * 6 * b * (a + epsilon * nodes) ** 2)
    assert finite == pytest.approx(integrated)
    assert abs(finite / epsilon - 6 * b * a * a) > 0.01


def test_post_pool_quadratic_has_compatible_symmetric_cross_response():
    rng = np.random.default_rng(168)
    a, b = rng.normal(size=(2, 3))
    m = rng.normal(size=(3, 3))
    m = (m + m.T) / 2
    da, db = rng.normal(size=(2, 3))
    epsilon = 1e-4
    grad_b = lambda x: m @ ((x + b) / 2) / 2
    response = db @ (
        (grad_b(a + epsilon * da) - grad_b(a - epsilon * da)) / (2 * epsilon)
    )
    assert response == pytest.approx(db @ m @ da / 4, abs=1e-10)
    assert response == pytest.approx(da @ m @ db / 4, abs=1e-10)
