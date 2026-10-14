"""Derivative, metric, compression and curvature checks for the explicit chart."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f10_geometry import (
    action_subspace,
    common_metric,
    edit_jacobian,
    project_edits,
    quadratic_response,
)


def test_chain_rule_with_nonsymmetric_quadratic():
    rng = np.random.default_rng(42)
    h0, a, u, b = (
        rng.normal(size=7),
        rng.normal(size=(7, 4)),
        rng.normal(size=(3, 7)),
        rng.normal(size=(3, 7, 7)),
    )
    e = rng.normal(size=4)
    exact = edit_jacobian(e, h0, a, u, b)
    for step in (1e-2, 1e-3, 1e-4):
        observed = np.column_stack(
            [
                (
                    quadratic_response(e + d, h0, a, u, b)
                    - quadratic_response(e - d, h0, a, u, b)
                )
                / (2 * step)
                for d in step * np.eye(4)
            ]
        )
        np.testing.assert_allclose(observed, exact, atol=1e-7)


def test_single_gradient_is_outer_product():
    j = np.array([1.0, 2, 3, 4])
    np.testing.assert_array_equal(common_metric(j), np.outer(j, j))
    for w in ([-1], [np.nan], [0], [1, 2]):
        with pytest.raises(ValueError):
            common_metric(j, w)


def test_generalized_eigen_and_nullspace():
    j = np.array([[1.0, 0, 0, 0], [0, 2, 0, 0]])
    c = np.diag([0.1, 0.2, 3, 9])
    result = action_subspace(j, c)
    v = result["basis"]
    np.testing.assert_allclose(
        result["metric"] @ v,
        result["denominator"] @ v @ np.diag(result["values"][:2]),
        atol=1e-12,
    )
    np.testing.assert_allclose(j @ result["nullspace"], 0)
    assert result["rank"] == 2
    edits = np.arange(20).reshape(5, 4)
    np.testing.assert_allclose(
        project_edits(edits, v, result["denominator"]) @ j.T, edits @ j.T, atol=1e-12
    )


def test_null_first_derivative_can_have_curvature():
    h0, a = np.zeros(4), np.eye(4)
    u = np.array([[1, 0, 0, 0]])
    b = np.zeros((1, 4, 4))
    b[0, 2, 2] = 2
    e = np.array([0, 0, 0.5, 0])
    assert (edit_jacobian(np.zeros(4), h0, a, u, b) @ e)[0] == 0
    assert quadratic_response(e, h0, a, u, b)[0] == 0.25


def test_invalid_covariance_rejected():
    with pytest.raises(ValueError):
        action_subspace(np.eye(4), -np.eye(4))


def test_zero_jacobian_full_nullspace():
    result = action_subspace(np.zeros((2, 4)), np.eye(4))
    assert result["rank"] == 0
    assert result["nullspace"].shape == (4, 4)
