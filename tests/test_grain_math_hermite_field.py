#!/usr/bin/env python3
"""Synthetic operator/unit/integrability tests for the HM26 potential compiler."""

import numpy as np
import pytest

from grain_research.construction import math_hermite_field as H


def fixture():
    """Create a deterministic curved source function with independent directions."""
    rng = np.random.default_rng(2601)
    x, v = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    y = np.sin(x[:, 0]) + x[:, 1] * x[:, 2]
    d = np.cos(x[:, 0]) * v[:, 0] + v[:, 1] * x[:, 2] + v[:, 2] * x[:, 1]
    return x, v, y, d


def test_kernel_derivative_finite_difference():
    x, v, _, _ = fixture()
    zero, one = np.zeros(8, int), np.ones(8, int)
    step = 1e-5
    analytic = H.covariance(x, v, one, x, v, zero, 2)
    fd = (
        H.covariance(x + step * v, v, zero, x, v, zero, 2)
        - H.covariance(x - step * v, v, zero, x, v, zero, 2)
    ) / (2 * step)
    np.testing.assert_allclose(analytic, fd, atol=2e-10)


def test_kernel_mixed_derivative_and_psd():
    x, v, _, _ = fixture()
    zero, one = np.zeros(8, int), np.ones(8, int)
    step = 1e-5
    analytic = H.covariance(x, v, one, x, v, one, 2)
    fd = (
        H.covariance(x, v, one, x + step * v, v, zero, 2)
        - H.covariance(x, v, one, x - step * v, v, zero, 2)
    ) / (2 * step)
    np.testing.assert_allclose(analytic, fd, atol=5e-10)
    model = H.fit_potential(x, v, np.zeros(8), np.ones(8), 2)
    np.testing.assert_allclose(model["gram"], model["gram"].T, atol=1e-14)
    assert np.linalg.eigvalsh(model["gram"]).min() > -1e-12


def test_augmented_direct_solve_and_constant_prior():
    x, v, y, d = fixture()
    m = H.fit_potential(x, v, y, d, 2)
    direct = np.linalg.solve(m["gram"] + np.diag(m["diagonal"] ** 2), m["target"])
    np.testing.assert_allclose(m["coefficients"], direct, atol=1e-14)
    shift = H.fit_potential(x, v, y + 5, d, 2)
    np.testing.assert_allclose(H.predict(shift, x) - H.predict(m, x), 5, atol=1e-14)
    np.testing.assert_allclose(H.predict(shift, x, v), H.predict(m, x, v), atol=1e-14)


def test_compiled_derivative_matches_potential():
    x, v, y, d = fixture()
    m = H.fit_potential(x, v, y, d, 2)
    step = 1e-5
    fd = (H.predict(m, x + step * v) - H.predict(m, x - step * v)) / (2 * step)
    np.testing.assert_allclose(fd, H.predict(m, x, v), atol=2e-10)


def test_scalar_potential_telescopes_and_derivative_odd():
    x, v, y, d = fixture()
    m = H.fit_potential(x, v, y, d, 2)
    f = H.predict(m, x[:3])
    assert abs((f[1] - f[0]) + (f[2] - f[1]) + (f[0] - f[2])) < 1e-14
    np.testing.assert_allclose(H.predict(m, x, -v), -H.predict(m, x, v), atol=1e-14)


def test_unit_transport_invariant_kernel():
    x, v, _, _ = fixture()
    kinds = np.ones(8, int)
    np.testing.assert_allclose(
        H.covariance(x, v, kinds, x, v, kinds, 2),
        H.covariance(3 * x + 9, 3 * v, kinds, 3 * x + 9, 3 * v, kinds, 18),
        atol=1e-13,
    )


def test_value_only_does_not_consume_derivative_targets():
    x, v, y, d = fixture()
    m1 = H.fit_potential(x, v, y, d, 2, hermite=False)
    m2 = H.fit_potential(x, v, y, d + 1000, 2, hermite=False)
    np.testing.assert_array_equal(m1["coefficients"], m2["coefficients"])


def test_source_geometry_and_invalid_inputs():
    x, v, _, _ = fixture()
    z, dv, origin, unit, length = H.source_geometry(x, v)
    np.testing.assert_allclose(z, (x - origin) / unit)
    np.testing.assert_allclose(dv, v / unit)
    assert length > 0
    with pytest.raises(ValueError):
        H.covariance(x, np.zeros_like(v), np.ones(8), x, v, np.ones(8), 2)
    with pytest.raises(ValueError):
        H.source_geometry(np.ones_like(x), v)
