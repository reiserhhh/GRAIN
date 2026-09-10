#!/usr/bin/env python3
"""Field orthogonality does not imply independent measurement errors."""

import numpy as np

from scripts.analyze_grain_content_coefficients import error_parts


def test_field_split_exact_but_action_components_can_cancel():
    j = np.eye(3)
    basis = np.eye(3)[:2]
    a = 3 * j @ basis.T
    a[0, 0] += 3
    d = np.ones((1, 3, 3))
    rows, missing, coefficient, predicted = error_parts(j, a, basis, d)
    np.testing.assert_allclose(rows["field_basis_error"], 1 / 3)
    np.testing.assert_allclose(rows["field_coefficient_error"], 1 / 3)
    np.testing.assert_allclose(missing, [-1])
    np.testing.assert_allclose(coefficient, [1])
    np.testing.assert_allclose(np.sum(predicted), np.sum(j))


def test_random_orthogonal_split_and_signed_action_identity():
    rng = np.random.default_rng(52)
    j = rng.normal(size=(30, 8))
    v = np.linalg.qr(rng.normal(size=(8, 3)))[0].T
    a = rng.normal(size=(30, 3))
    d = rng.normal(size=(4, 30, 8))
    row, b, c, pred = error_parts(j, a, v, d)
    assert abs(row["field_split_error"]) < 1e-12
    np.testing.assert_allclose(np.einsum("th,kth->k", pred - j, d), b + c, atol=1e-12)
