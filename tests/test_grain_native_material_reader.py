#!/usr/bin/env python3
"""Numerical witnesses for the common material / localized reader distinction."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.native_material_reader import (
    allocate_slices,
    common_prefix_length,
    direction,
    tangent_parts,
)


def sample():
    """Generate deterministic independent matrices, not human data."""
    rng = np.random.default_rng(18)
    return rng.normal(size=(13, 6)), rng.normal(size=(13, 4))


def test_prefix_suffix_and_early_query():
    assert common_prefix_length([[1, 2, 3, 5], [1, 2, 3, 8, 9]]) == 3
    assert common_prefix_length([[1, 5, 2, 3], [1, 8, 2, 3]]) == 1


def test_fractional_boundary_allocation():
    pos, g, weights = allocate_slices(
        [(0, 1), (1, 4), (4, 6), (6, 8)], [(1, 3), (3, 6)], [[2], [8]]
    )
    np.testing.assert_array_equal(pos, [1, 2])
    np.testing.assert_allclose(g[:, 0], [4, 8])
    np.testing.assert_allclose(weights.sum(1), 1)


def test_partial_excluded_tokens_are_not_retained():
    pos, _, _ = allocate_slices([(0, 2), (2, 3), (3, 4), (4, 6)], [(1, 4)], [[2]])
    np.testing.assert_array_equal(pos, [1, 2])


def test_reject_overlapping_slices():
    with pytest.raises(ValueError):
        allocate_slices([(1, 3), (3, 4)], [(1, 3), (2, 4)], [[1], [2]])


def test_constant_reader_is_covariance_reader():
    h, g = sample()
    w = np.arange(h.shape[1], dtype=float)
    p = tangent_parts(h, g, np.tile(w / len(h), (len(h), 1)))
    np.testing.assert_allclose(p["raw_total"], p["operator"] @ w)
    np.testing.assert_allclose(p["raw_local"], 0, atol=1e-14)


def test_local_reader_survives_zero_average_reader():
    h = np.array([[-1.0], [0.0], [1.0]])
    g = np.array([[1.0], [-2.0], [1.0]])
    p = tangent_parts(h, g, h)
    assert p["raw_common"][0] == 0
    assert p["raw_total"][0] == 2
    assert p["raw_local"][0] == 2


def test_raw_offset_and_paired_permutation_invariance():
    h, g = sample()
    j = h**2 + 0.5
    a = tangent_parts(h, g, j)
    b = tangent_parts(h[::-1] + 3, g[::-1] - 4, j[::-1])
    for key in ("raw_total", "raw_common", "raw_local", "operator"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-13)
    assert a["identity_error"] < 1e-13


def test_nonlinear_finite_difference():
    h, g = sample()
    target = lambda z: np.sin(np.sum(z**2) / 20)
    j = np.cos(np.sum(h**2) / 20) * h / 10
    p = tangent_parts(h, g, j)
    for k in range(4):
        d, _ = direction(h, g, k)
        eps = 1e-6
        actual = (target(h + eps * d) - target(h - eps * d)) / (2 * eps)
        np.testing.assert_allclose(actual, p["total"][k], atol=2e-9)


def test_zero_direction_has_zero_response():
    h, _ = sample()
    g = np.ones((len(h), 4))
    d, scale = direction(h, g, 0)
    assert not np.any(d)
    assert scale == 0
    assert not np.any(tangent_parts(h, g, h)["total"])


def test_invalid_jacobian_rejected():
    h, g = sample()
    with pytest.raises(ValueError):
        tangent_parts(h, g, h[:-1])
