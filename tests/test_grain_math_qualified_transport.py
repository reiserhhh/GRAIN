#!/usr/bin/env python3
"""Synthetic arithmetic tests for qualified-evidence transport, not validity."""

import numpy as np
import pytest

from grain_research.construction.math_qualified_transport import (
    SIGNS,
    marginal_means,
    paired_profile_contrast,
    principal_cosines,
    project_joint_corners,
    vector_cosine,
)
from grain_research.construction.math_text_response import support_frame


def test_mixed_projection_keeps_both_marginals():
    rng = np.random.default_rng(105)
    x = rng.normal(size=(4, 20))
    b, _ = support_frame(rng.normal(size=(20, 6)))
    y, kept, residual = project_joint_corners(x, b)
    np.testing.assert_allclose(SIGNS @ y, kept, atol=1e-13)
    np.testing.assert_allclose(marginal_means(x), marginal_means(y), atol=1e-13)
    np.testing.assert_allclose(kept + residual, SIGNS @ x, atol=1e-13)


def test_principal_angles_ignore_axis_rotations():
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=(25, 6)), rng.normal(size=(25, 6))
    u, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    v, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    np.testing.assert_allclose(principal_cosines(a, b), principal_cosines(a @ u, b @ v))
    np.testing.assert_allclose(principal_cosines(a, a @ u), np.ones(6), atol=1e-13)


def test_shared_span_does_not_require_shared_axes():
    a = np.eye(12)[:, :6]
    b = a[:, ::-1]
    np.testing.assert_allclose(principal_cosines(a, b), 1)
    assert vector_cosine(a[:, 0], b[:, 0]) == 0
    np.testing.assert_allclose(principal_cosines(a, np.eye(12)[:, 6:]), 0)


def test_actual_profile_contrast_and_equal_marginals():
    f = np.arange(24).reshape(4, 6) ** 2
    np.testing.assert_allclose(paired_profile_contrast(f), 0.25 * SIGNS @ f)
    plus, minus = (
        (1 + 0.5 * SIGNS).reshape(2, 2) / 4,
        (1 - 0.5 * SIGNS).reshape(2, 2) / 4,
    )
    np.testing.assert_allclose(plus.sum(0), minus.sum(0))
    np.testing.assert_allclose(plus.sum(1), minus.sum(1))
    assert np.min(plus) > 0 and np.min(minus) > 0


def test_source_local_mixed_projection_identity():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(4, 30))
    b, _ = support_frame((SIGNS @ x)[:, None])
    y, _, residual = project_joint_corners(x, b)
    np.testing.assert_allclose(y, x, atol=1e-13)
    np.testing.assert_allclose(residual, 0, atol=1e-13)


def test_bad_support_and_corners_fail_explicitly():
    with pytest.raises(ValueError):
        principal_cosines(np.zeros((4, 2)), np.ones((4, 1)))
    with pytest.raises(ValueError):
        principal_cosines(np.ones((4, 2)), np.ones((5, 2)))
    with pytest.raises(ValueError):
        project_joint_corners(np.ones((4, 8)), np.ones((8, 2)))
    with pytest.raises(ValueError):
        paired_profile_contrast(np.ones((3, 2)))
