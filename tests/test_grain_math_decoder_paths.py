#!/usr/bin/env python3
"""Exact polynomial, candidate-readout, and path-dependence checks for HM25."""

import numpy as np
import pytest

from grain_research.construction.math_decoder_paths import (
    candidate_score_gradient,
    finite_account,
    integrate,
    simpson_weights,
)


def test_polynomial_path_integral():
    """A finite path captures an effect missed by the zero start derivative."""
    t = np.linspace(0, 1, 9)
    assert np.isclose(integrate(t, 3 * t**2), 1)
    assert np.isclose(integrate(t, 4 * t**3), 1)
    assert np.isclose(simpson_weights(t).sum(), 1)


def test_vector_covector_contraction():
    """Integrating a covector and contracting commute for a fixed displacement."""
    t = np.linspace(0, 1, 9)
    g = np.stack((t, t**2), axis=1)
    d = np.array([2.0, 3.0])
    assert np.isclose(integrate(t, g) @ d, integrate(t, g @ d))


def test_endpoint_total_not_unique_component_attribution():
    """The exact differential has one total but different component path shares."""
    t = np.linspace(0, 1, 9)
    straight = np.array([integrate(t, t), integrate(t, t)])
    x_then_y = np.array([0.0, 1.0])
    assert np.isclose(straight.sum(), x_then_y.sum())
    assert not np.allclose(straight, x_then_y)


def test_readout_gradient_and_constant_logit_gauge():
    """Candidate covectors annihilate the additive logit gauge."""
    z, a = np.array([0.3, -0.1, 0.7]), np.array([0.0, 0.5, 1.0])
    score, gradient = candidate_score_gradient(z, a)
    eps = 1e-5
    for i in range(3):
        d = np.eye(3)[i] * eps
        finite = (
            candidate_score_gradient(z + d, a)[0]
            - candidate_score_gradient(z - d, a)[0]
        ) / (2 * eps)
        assert np.isclose(finite, gradient[i], atol=1e-10)
    assert abs(gradient.sum()) < 1e-14
    assert np.isclose(candidate_score_gradient(z + 17, a)[0], score)


def test_error_account_identity():
    """Do not label all endpoint-minus-tangent error as readout curvature."""
    result = finite_account(
        np.array([0.1, 0.2]), np.array([-0.2, 0.8]), np.array([0.0, 1.0]), -0.01
    )
    assert np.isclose(
        result["decoder_nonlinear_increment"] + result["readout_nonlinear_increment"],
        result["full_nonlinear_increment"],
    )


@pytest.mark.parametrize("nodes", [[0, 0.5], [0, 0.2, 1], [1, 0.5, 0]])
def test_invalid_quadrature(nodes):
    """Reject unplanned or nonuniform grids rather than silently extrapolating."""
    with pytest.raises(ValueError):
        simpson_weights(nodes)
