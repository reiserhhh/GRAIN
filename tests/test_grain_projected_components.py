#!/usr/bin/env python3
"""Common projection tests for conditional anchor and transport accounts."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.projected_components import (
    constant_fiber_loss,
    projected_component_draws,
    projected_components,
)


def test_identical_input_with_varying_targets_has_exact_constant_risk_floor():
    target = np.array([-0.3, 0.1, 0.2, 0.7])
    for constant in [-10, 0, target.mean(), 5]:
        result = constant_fiber_loss(target, constant)
        assert result["fixed_constant_mse"] >= result["minimum_constant_mse"] - 1e-14
        assert result["identity_error"] < 1e-13
    assert constant_fiber_loss(target, target.mean())["squared_mean_bias"] == 0


def sample():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(40, 3))
    q, _ = np.linalg.qr(x - x.mean(0))
    return q.T


def test_common_projector_preserves_signed_cancellation():
    x, y, _ = sample()
    parts = np.column_stack([x + 2 * y, x, -y])
    a = projected_components(2 * x + y, parts, x, y)
    np.testing.assert_allclose(
        a["contributions"], [0, 2 / np.sqrt(5), 0, -1 / np.sqrt(5)], atol=1e-14
    )
    assert a["closure_error"] < 1e-14
    assert a["orthogonality_error"] < 1e-14


def test_batched_projectors_are_recomputed_per_author_resample():
    x, y, z = sample()
    parts = np.column_stack([x + y, z, -0.4 * y])
    t = parts.sum(1)
    ix = np.random.default_rng(2).integers(0, len(x), (7, len(x)))
    batched = projected_component_draws(t, parts, x, y, ix)
    for i, j in enumerate(ix):
        a = projected_components(t[j], parts[j], x[j], y[j])
        np.testing.assert_allclose(batched[i], a["contributions"], atol=1e-14)


def test_target_parts_must_reconstruct_and_proxy_must_vary():
    x, y, z = sample()
    with pytest.raises(ValueError):
        projected_components(x, np.column_stack([y, z]), x, y)
    with pytest.raises(ValueError):
        projected_components(x, x[:, None], np.ones(len(x)), y)


def test_conditioned_integral_reconstructs_a_nonlinear_function():
    """A constant Jacobian can fail while the exact conditional path closes."""
    e = np.linspace(-1, 2, 17)
    x = np.linspace(0.1, 1.3, 17)
    x0 = -0.4
    f = lambda a, b: a**2 + a * b**2 + b**3
    nodes, weights = np.polynomial.legendre.leggauss(3)
    path = x0 + (nodes[:, None] + 1) / 2 * (x - x0)
    k = (weights / 2) @ (2 * e * path + 3 * path**2)
    np.testing.assert_allclose(f(e, x), f(e, x0) + k * (x - x0), atol=1e-14)
    fixed = f(0.0, x0) + 3 * x0**2 * (x - x0)
    gap = f(e, x0) - f(0.0, x0) + (k - 3 * x0**2) * (x - x0)
    np.testing.assert_allclose(f(e, x) - fixed, gap, atol=1e-14)


def test_reanchoring_changes_component_covariance_not_total_function():
    """Exact decomposition does not identify a reference-independent component."""
    e = np.linspace(-1, 1, 15)
    x = np.sin(e)
    f = lambda a, b: a**2 + a * b + b**2
    b0, b1, total = f(e, 0.0), f(e, 1.0), f(e, x)
    r0, r1, shift = total - b0, total - b1, b1 - b0
    np.testing.assert_allclose(r1, r0 - shift, atol=1e-14)
    np.testing.assert_allclose(b0 + r0, b1 + r1, atol=1e-14)
    cov = lambda a: float(np.mean((a - a.mean()) * e))
    assert abs(cov(b1) - cov(b0)) > 0.2
    assert abs(cov(b1) + cov(r1) - cov(total)) < 1e-14
