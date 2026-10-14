#!/usr/bin/env python3
"""Falsifiable gradient, connection, loop and linear-control tests."""

import numpy as np
import pytest

from grain_research.construction.math_response_connection import (
    AttentionResponse,
    horizontal,
    horizontal_derivative,
    rectangle,
)


def model(mode="joint"):
    """Produce a small nontrivial GQA fixture unrelated to scientific outcomes."""
    rng = np.random.default_rng(103)
    return AttentionResponse(
        rng.normal(size=(4, 2, 3)),
        rng.normal(size=(4, 2, 3)),
        rng.normal(size=(3, 4, 3)),
        rng.normal(size=(3, 4, 3)),
        mode,
    )


def test_attention_gradient_and_hessian_direction():
    """Central differences independently check analytic first and second derivatives."""
    m = model()
    rng = np.random.default_rng(1)
    a = rng.normal(size=m.x.shape)
    h = 1e-5 / np.linalg.norm(a)
    _, j = m.value_jacobian(m.x)
    fp, jp = m.value_jacobian(m.x + h * a)
    fm, jm = m.value_jacobian(m.x - h * a)
    np.testing.assert_allclose((fp - fm) / (2 * h), j @ a, atol=1e-8)
    np.testing.assert_allclose(
        (jp - jm) / (2 * h), m.directional_jacobian(m.x, a), atol=1e-8
    )


def test_dual_controller_and_vertical_bracket():
    """Independent coordinate controls can have a nonzero vertical commutator."""
    m = model()
    _, j, u, c, pairs, _ = m.brackets(m.x)
    np.testing.assert_allclose(j @ u, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(j @ c.T, 0, atol=1e-11)
    assert np.linalg.norm(c) > 1e-3
    i, k = pairs[0]
    h = 1e-5
    di = (m.connection(m.x + h * u[:, i])[2] - m.connection(m.x - h * u[:, i])[2]) / (
        2 * h
    )
    dk = (m.connection(m.x + h * u[:, k])[2] - m.connection(m.x - h * u[:, k])[2]) / (
        2 * h
    )
    np.testing.assert_allclose(di[:, k] - dk[:, i], c[0], atol=1e-8)


def test_rectangle_area_scaling_orientation_and_output_closure():
    """A score loop is closed while internal drift follows the bracket sign."""
    m = model()
    f, _, _, c, _, _ = m.brackets(m.x)
    eps = 0.002
    big = rectangle(m, m.x, 0, 1, eps) - m.x
    small = rectangle(m, m.x, 0, 1, eps / 2) - m.x
    reverse = rectangle(m, m.x, 1, 0, eps) - m.x
    assert np.linalg.norm(big / eps**2 - c[0]) / np.linalg.norm(c[0]) < 0.01
    assert abs(np.linalg.norm(big) / np.linalg.norm(small) - 4) < 0.02
    assert np.linalg.norm(big + reverse) / np.linalg.norm(big) < 0.01
    np.testing.assert_allclose(m.value_jacobian(m.x + big)[0], f, atol=1e-9)


def test_value_only_has_no_curvature_or_internal_loop_drift():
    """Linear-in-V fields commute rather than manufacturing curvature numerically."""
    m = model("value_only")
    _, _, _, c, _, _ = m.brackets(m.x)
    np.testing.assert_array_equal(c, 0)
    np.testing.assert_allclose(rectangle(m, m.x, 0, 1, 0.002), m.x, atol=1e-13)


def test_common_key_gauge_leaves_connection_and_brackets_invariant():
    """Every target function, its tangent geometry and controller respect key shifts."""
    m = model()
    gauge = np.array([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]])
    n = AttentionResponse(m.k + gauge, m.v, m.q, m.d)
    for a, b in zip(m.brackets(m.x)[:4], n.brackets(n.x)[:4], strict=True):
        np.testing.assert_allclose(a, b, atol=1e-11)


def test_known_toy_bracket_and_affine_control():
    """F=(x,y+xz) has vertical bracket (0,0,1) at the origin; affine maps do not."""
    j = np.array([[1.0, 0, 0], [0, 1, 0]])
    u, inv, _ = horizontal(j)
    hx = np.array([[0.0, 0, 0], [0, 0, 1]])
    hy = np.zeros_like(j)
    dx = horizontal_derivative(j, hx, u, inv)
    dy = horizontal_derivative(j, hy, u, inv)
    np.testing.assert_array_equal(dx[:, 1] - dy[:, 0], [0, 0, 1])
    np.testing.assert_array_equal(horizontal_derivative(j, hy, u, inv), 0)


def test_rank_loss_and_invalid_geometry_do_not_get_regularized():
    """Undefined fields or malformed carriers fail, without a fallback direction."""
    with pytest.raises(ValueError, match="rank loss"):
        horizontal(np.array([[1.0, 0], [2, 0]]))
    m = model()
    with pytest.raises(ValueError):
        m.value_jacobian(np.zeros(3))
    with pytest.raises(ValueError):
        rectangle(m, m.x, 0, 1, 0)
