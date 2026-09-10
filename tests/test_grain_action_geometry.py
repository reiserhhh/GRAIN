#!/usr/bin/env python3
"""Witnesses for support-limited corrections, output geometry and their limits."""

import numpy as np

from grain_research.vector_mechanisms.action_geometry import (
    axis_account,
    fit_support_correction,
    output_gradient,
    read_support,
    soft_support,
)
from grain_research.vector_mechanisms.comment_carrier import score


def test_support_fit_matches_primal_and_stationarity():
    """Dual evaluation solves the stated prior-centered objective, not ordinary shrinkage."""
    rng = np.random.default_rng(1)
    d, y, g = rng.normal(size=(4, 7)), rng.normal(size=4), rng.normal(size=7)
    model = fit_support_correction(d, y, g)
    lam = model["regularization"]
    expected = np.linalg.solve(d.T @ d + lam * np.eye(7), d.T @ y + lam * g)
    got = g + model["correction"]
    np.testing.assert_allclose(got, expected, atol=1e-12)
    np.testing.assert_allclose(d.T @ (d @ got - y) + lam * (got - g), 0, atol=1e-12)


def test_null_support_and_zero_input_preserve_prior():
    """Unseen orthogonal directions cannot be learned from zero reference response."""
    d = np.array([[1.0, 0, 0], [-1, 0, 0]])
    g = np.array([0.2, 0.4, 0.6])
    m = fit_support_correction(d, [2, -2], g)
    x = np.array([[0.0, 1, 0], [0, 0, 0]])
    np.testing.assert_allclose(read_support(x, m), x @ g)
    np.testing.assert_allclose(soft_support(x, m), 0)


def test_rotation_and_unit_invariance():
    """A common orthogonal chart or input-unit change does not alter the action."""
    rng = np.random.default_rng(2)
    d, x, y, g = (
        rng.normal(size=(4, 7)),
        rng.normal(size=(3, 7)),
        rng.normal(size=4),
        rng.normal(size=7),
    )
    u = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    m = fit_support_correction(d, y, g)
    rotated = fit_support_correction(d @ u, y, u.T @ g)
    scaled = fit_support_correction(3 * d, y, g / 3)
    np.testing.assert_allclose(
        read_support(x, m), read_support(x @ u, rotated), atol=1e-12
    )
    np.testing.assert_allclose(
        read_support(x, m), read_support(3 * x, scaled), atol=1e-12
    )


def test_held_targets_and_soft_support():
    """Only reference effects set the map, and support energy stays within[0,1]."""
    rng = np.random.default_rng(3)
    d, y = rng.normal(size=(9, 6)), rng.normal(size=9)
    g = np.zeros(6)
    a = fit_support_correction(d[:4], y[:4], g)
    y[4:] += 999
    b = fit_support_correction(d[:4], y[:4], g)
    np.testing.assert_array_equal(a["correction"], b["correction"])
    s = soft_support(d, a)
    assert (s >= -1e-12).all() and (s <= 1 + 1e-12).all()


def test_axis_orthogonal_energy_closes():
    """Error in shared direction and orthogonal directions exhausts the same error."""
    rng = np.random.default_rng(4)
    y, p = rng.normal(size=(2, 8, 5))
    v = rng.normal(size=5)
    v /= np.linalg.norm(v)
    a = axis_account(y, p, v)
    np.testing.assert_allclose(a["closure_error"], 0, atol=1e-12)
    np.testing.assert_allclose(
        a["parallel_energy"] + a["actual_perpendicular_norm"] ** 2, a["actual_energy"]
    )


def test_output_gradient_matches_difference_and_common_shift():
    """The output-aware direction is analytic, not picked from personality labels."""
    z, v = np.array([0.2, -0.5, 0.8]), np.array([0.0, 0.5, 1.0])
    g = output_gradient(z, v)
    h = 1e-5
    numerical = np.array(
        [
            (score(z + np.eye(3)[i] * h, v) - score(z - np.eye(3)[i] * h, v)) / (2 * h)
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(g, numerical, atol=1e-10)
    assert abs(g.sum()) < 1e-12


def test_target_projection_commutes_with_fixed_kernel_reading():
    """Changing to a linear scalar target alone cannot improve the same kernel map."""
    rng = np.random.default_rng(5)
    x, test, y, v = (
        rng.normal(size=(9, 4)),
        rng.normal(size=(2, 4)),
        rng.normal(size=(9, 3)),
        rng.normal(size=3),
    )
    k = x @ x.T + 0.1 * np.eye(9)
    full = test @ x.T @ np.linalg.solve(k, y)
    scalar = test @ x.T @ np.linalg.solve(k, y @ v)
    np.testing.assert_allclose(full @ v, scalar, atol=1e-12)


def test_common_gradient_energy_does_not_bound_relative_action_error():
    """Bounded input can miss all nonzero action even when gradients are almost identical."""
    g0, delta, d = np.array([1.0, 0]), np.array([0.0, 1e-4]), np.array([0.0, 1.0])
    assert g0 @ g0 / ((g0 + delta) @ (g0 + delta)) > 0.99999
    assert g0 @ d == 0 and (g0 + delta) @ d == 1e-4
