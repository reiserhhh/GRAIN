#!/usr/bin/env python3
"""Synthetic witnesses for binary-link and derivative-surrogate algebra."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.retained_logit import (
    fit_curvature,
    link_account,
    logit,
    path_account,
    quadratic_delta,
    quadrature,
    radial_parts,
    sigmoid,
)


def test_logistic_linear_has_only_link_curvature():
    a, b = 1.4, 0.7
    p0, p1 = sigmoid([a, a + b])
    account = link_account(p0, p1, p0 * (1 - p0) * b)
    assert abs(account["internal_remainder"]) < 1e-15
    assert abs(account["logit_remainder"]) < 1e-15
    assert abs(account["link_curvature"]) > 0.01
    assert account["identity_error"] < 1e-15


def test_nonlinear_logit_internal_remainder_and_exact_identity():
    a, b, c = -0.2, 0.4, -0.9
    p0, p1 = sigmoid([a, a + b + c])
    result = link_account(p0, p1, p0 * (1 - p0) * b)
    assert result["logit_remainder"] == pytest.approx(c)
    assert result["internal_remainder"] == pytest.approx(p1 - sigmoid(a + b))
    assert result["identity_error"] < 1e-15


def test_path_chain_rule_and_logit_quadrature():
    t = np.linspace(0, 1, 5)
    p = sigmoid(0.3 + 0.5 * t + 0.7 * t**2)
    v = (0.5 + 1.4 * t)[:, None, None]
    g = (p * (1 - p))[:, None, None] * v
    result, recovered = path_account(t, p, g, np.ones((1, 1)))
    np.testing.assert_allclose(recovered, v)
    assert result["logit_integration_error"] < 1e-15
    assert result["logit_remainder"] == pytest.approx(0.7)


@pytest.mark.parametrize("p", [0, 1, -0.1, 1.1, np.nan, np.inf])
def test_invalid_probabilities_are_not_clipped(p):
    with pytest.raises(ValueError):
        logit(p)


def test_quadrature_rejects_arbitrary_nodes():
    with pytest.raises(ValueError):
        quadrature([0, 0.3, 1], [1, 2, 3])
    with pytest.raises(ValueError):
        path_account([0, 0.5, 1], [0.5] * 3, np.ones((3, 2, 2)), np.ones((1, 2)))


def test_derivative_fit_recovers_known_quadratic_and_shrinkage():
    rng = np.random.default_rng(71)
    ds = rng.normal(size=(4, 3, 4))
    e = rng.normal(size=(3, 4))
    e /= np.linalg.norm(e, axis=1)[:, None]
    v0 = rng.normal(size=(3, 4))
    k = rng.normal(size=(2, 3))
    ts = [np.linspace(0, 1, n) for n in (3, 5, 3, 5)]
    vs = []
    for d, t in zip(ds, ts, strict=True):
        par, perp = radial_parts(d, e)
        kd = k[0, :, None] * par + k[1, :, None] * perp
        vs.append(v0 + t[:, None, None] * kd)
    mean, coef, _ = fit_curvature(ds, np.broadcast_to(e, ds.shape), ts, vs, penalty=0)
    np.testing.assert_allclose(coef, k, atol=1e-14)
    np.testing.assert_allclose(mean, v0)
    _, shrunk, _ = fit_curvature(ds, np.broadcast_to(e, ds.shape), ts, vs)
    np.testing.assert_allclose(shrunk, k / 1.01)
    d = ds[0]
    linear, full = quadratic_delta(d, e, mean, coef)
    expected = 0.5 * sum(
        np.sum(k[j] * np.sum(x**2, axis=1)) for j, x in enumerate(radial_parts(d, e))
    )
    assert full - linear == pytest.approx(expected)


def test_zero_displacement_and_rotation_invariance():
    rng = np.random.default_rng(22)
    d, v = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
    e = rng.normal(size=(2, 3))
    e /= np.linalg.norm(e, axis=1)[:, None]
    k = rng.normal(size=(2, 2))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert quadratic_delta(np.zeros_like(d), e, v, k) == (0.0, 0.0)
    np.testing.assert_allclose(
        quadratic_delta(d, e, v, k), quadratic_delta(d @ q, e @ q, v @ q, k)
    )


def test_radial_frame_and_fit_fail_explicitly():
    with pytest.raises(ValueError):
        radial_parts(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        fit_curvature(np.zeros((0, 2, 3)), np.zeros((0, 2, 3)), [], [])


def test_stable_sigmoid_extremes():
    np.testing.assert_allclose(sigmoid([-1000, 0, 1000]), [0, 0.5, 1])


def test_radial_diagonal_surrogate_can_miss_cross_coordinate_action():
    """Off-diagonal curvature is invisible to a pooled diagonal fit on this design."""
    d = np.array([[[1, 1]], [[1, -1]], [[-1, 1]], [[-1, -1]]], float)
    e = np.broadcast_to(np.array([[1.0, 0.0]]), d.shape)
    hessian = np.array([[0.0, 1.0], [1.0, 0.0]])
    t = np.linspace(0, 1, 3)
    gradients = [t[:, None, None] * (di @ hessian) for di in d]
    origin, k, _ = fit_curvature(d, e, [t] * 4, gradients, penalty=0)
    np.testing.assert_allclose(k, 0, atol=1e-15)
    actual = np.array([0.5 * np.sum((di @ hessian) * di) for di in d])
    predicted = np.array(
        [quadratic_delta(di, ei, origin, k)[1] for di, ei in zip(d, e, strict=True)]
    )
    np.testing.assert_allclose(actual, [1, -1, -1, 1])
    np.testing.assert_allclose(predicted, 0)


def test_gradient_energy_and_directional_action_are_distinct():
    """Large field changes orthogonal to a displacement have zero path action."""
    d = np.array([[1.0, 0.0]])
    field = np.array([[0.0, 1000.0]])
    assert np.linalg.norm(field) == 1000
    assert np.sum(field * d) == 0
