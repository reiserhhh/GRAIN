#!/usr/bin/env python3
"""Actual-corner, secant, metric and pulled-back response controls."""

import numpy as np
import pytest

from grain_research.construction.math_response_connection import AttentionResponse
from grain_research.construction.math_text_response import (
    TextSpanResponse,
    innovation,
    mixed_contrast,
    support_frame,
)


def fixture(mode="joint"):
    """Return a small frozen attention law, not scientific outcome data."""
    rng = np.random.default_rng(104)
    return AttentionResponse(
        rng.normal(size=(4, 1, 3)),
        rng.normal(size=(4, 1, 3)),
        rng.normal(size=(2, 2, 3)),
        rng.normal(size=(2, 2, 3)),
        mode,
    )


def test_mixed_text_contrast_detects_direction_outside_marginals():
    """Content and owner can have no marginal third coordinate yet a mixed one."""
    a, b, w = np.eye(3)
    corners = [a + b + w, -a + b - w, a - b - w, -a - b + w]
    delta = mixed_contrast(*corners)
    np.testing.assert_array_equal(delta, 4 * w)
    basis, singular = support_frame(np.stack([a, b], axis=1))
    p, r, ratio = innovation(delta, basis)
    np.testing.assert_array_equal(p, 0)
    np.testing.assert_array_equal(r, delta)
    assert ratio == 1 and len(singular) == 2
    np.testing.assert_array_equal(mixed_contrast(a + b, -a + b, a - b, -a - b), 0)


def test_empirical_frame_retains_rank_not_column_count():
    """Duplicated and zero secants are not rescued by ridge or fabricated directions."""
    basis, s = support_frame(np.array([[1, 2, 0], [0, 0, 0]], float))
    assert basis.shape == (2, 1) and s[-1] == 0
    empty, _ = support_frame(np.zeros((4, 7)))
    assert empty.shape == (4, 0)
    with pytest.raises(ValueError):
        support_frame(np.full((2, 3), np.nan))


def test_chart_chain_rule_and_hessian():
    """Central differences check the same function used by the response fields."""
    m = fixture()
    rng = np.random.default_rng(4)
    b, _ = support_frame(rng.normal(size=(len(m.x), 3)))
    c = TextSpanResponse(m, b)
    a = np.array([0.2, -0.3, 0.4])
    h = 1e-5
    _, j = c.value_jacobian(c.x)
    fp, jp = c.value_jacobian(c.x + h * a)
    fm, jm = c.value_jacobian(c.x - h * a)
    np.testing.assert_allclose((fp - fm) / (2 * h), j @ a, atol=1e-9)
    np.testing.assert_allclose(
        (jp - jm) / (2 * h), c.directional_jacobian(c.x, a), atol=1e-9
    )
    _, _, u, brackets, _, _ = c.brackets(c.x)
    np.testing.assert_allclose(j @ u, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(j @ brackets.T, 0, atol=1e-10)


def test_square_text_chart_is_integrability_control():
    """A full-rank square readout is locally a coordinate change, not curved fiber."""
    m = fixture()
    _, j = m.value_jacobian(m.x)
    b, _ = support_frame(j.T)
    c = TextSpanResponse(m, b)
    _, _, _, brackets, _, _ = c.brackets(c.x)
    np.testing.assert_allclose(brackets, 0, atol=1e-10)


def test_value_only_text_chart_is_linear():
    """Actual text V secants alone do not create nonlinear response curvature."""
    m = fixture("value_only")
    rng = np.random.default_rng(8)
    b, _ = support_frame(rng.normal(size=(len(m.x), 3)))
    c = TextSpanResponse(m, b)
    np.testing.assert_array_equal(c.brackets(c.x)[3], 0)


def test_metric_change_and_insufficient_support_fail_explicitly():
    """No whitening or deficient observation chart silently changes the object."""
    m = fixture()
    with pytest.raises(ValueError):
        TextSpanResponse(m, np.ones((len(m.x), 3)))
    b, _ = support_frame(np.ones((len(m.x), 1)))
    c = TextSpanResponse(m, b)
    with pytest.raises(ValueError, match="rank loss"):
        c.connection(c.x)


def test_complete_actual_corner_support():
    """Every content-by-owner corner exists without an outcome-selected subset."""
    from scripts.run_grain_text_response_hm104 import design

    rows = list(design())
    assert len(rows) == len(set(rows)) == 39
    assert (-1, 1, 1) in rows
    for axis in range(6):
        for level in (0, 2):
            assert (axis, level, 1) in rows
            for owner in (0, 2):
                assert (axis, level, owner) in rows
