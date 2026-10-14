#!/usr/bin/env python3
"""Complete isotropic/traceless accounts, not a new psychometric validation."""

import numpy as np

from scripts.describe_grain_condition_shape_hm187 import split


def test_trace_traceless_inner_product_and_orthogonal_frames():
    rng = np.random.default_rng(19)
    a, b = rng.normal(size=(2, 5, 5))
    a, b = a + a.T, b + b.T
    ai, at = split(a)
    bi, bt = split(b)
    np.testing.assert_allclose(np.trace(at), 0, atol=1e-14)
    np.testing.assert_allclose(np.sum(a * b), np.sum(ai * bi) + np.sum(at * bt))
    np.testing.assert_allclose(np.sum(ai * bi), np.trace(a) * np.trace(b) / 5)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    qi, qt = split(q.T @ a @ q)
    np.testing.assert_allclose(qi, q.T @ ai @ q, atol=1e-14)
    np.testing.assert_allclose(qt, q.T @ at @ q, atol=1e-14)


def test_pure_scalar_and_pure_anisotropic_both_retained():
    a = np.diag([-2.0, 1.0, 1.0])
    ai, at = split(a)
    np.testing.assert_array_equal(ai, 0)
    np.testing.assert_array_equal(at, a)
    bi, bt = split(-3 * np.eye(3))
    np.testing.assert_array_equal(bi, -3 * np.eye(3))
    np.testing.assert_array_equal(bt, 0)
