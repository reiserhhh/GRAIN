#!/usr/bin/env python3
"""Exact program decompositions do not require text models or private data."""

import numpy as np
import pytest

from grain_research.construction.math_evidence_diagnostics import coordinate_attribution


def test_multilinear_sum_and_signed_cancellation():
    a = np.array([[1, 2, 3], [-1, 1, 0]], float)
    b = np.array([[2, -3, 4], [2, 0, 1]], float)
    f = lambda z: z[:, 0] + z[:, 0] * z[:, 1] + z[:, 0] * z[:, 1] * z[:, 2]
    phi = coordinate_attribution(a, b, f)
    np.testing.assert_allclose(phi.sum(axis=1), f(b) - f(a), atol=1e-12)
    assert (phi < 0).any()


def test_linear_function_no_false_interaction():
    a = np.zeros((2, 3))
    b = np.array([[1, 2, 3], [2, 1, -1]], float)
    w = np.array([1, -2, 4])
    np.testing.assert_allclose(coordinate_attribution(a, b, lambda z: z @ w), b * w)


def test_unchanged_coordinate_contributes_zero():
    a = np.array([[1, 2]])
    b = np.array([[1, 3]])
    phi = coordinate_attribution(a, b, lambda z: z[:, 0] * z[:, 1])
    np.testing.assert_allclose(phi, [[0, 1]])


def test_bad_shape_rejected():
    with pytest.raises(ValueError):
        coordinate_attribution(np.zeros((1, 2)), np.ones((1, 3)), lambda z: z[:, 0])
