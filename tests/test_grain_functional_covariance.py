#!/usr/bin/env python3
"""Signed shared-path accounting is not an independent variance partition."""

import numpy as np
import pytest

from scripts.analyze_grain_functional_covariance import covariance_ledger


def test_signed_covariance_and_translation():
    r = np.random.default_rng(52)
    a, b = r.normal(size=(48, 4)), r.normal(size=(48, 4))
    matrix, total, error = covariance_ledger(a, b)
    assert error < 1e-12
    assert total == pytest.approx(np.cov(a.sum(1), b.sum(1), ddof=0)[0, 1])
    other, _, _ = covariance_ledger(a + 30, b - 42)
    np.testing.assert_allclose(matrix, other, atol=1e-12)


def test_cancellation_preserved():
    x = np.arange(12.0)
    a = np.column_stack([x, -x])
    matrix, total, error = covariance_ledger(a, a)
    assert matrix[0, 0] > 0 and matrix[0, 1] < 0
    assert total == 0 and error < 1e-12
