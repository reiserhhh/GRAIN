#!/usr/bin/env python3
"""Common-denominator covariance, including legitimate signed cancellation."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.criterion_accounting import criterion_account


def test_signed_parts_are_not_variance_shares():
    y = np.arange(10, dtype=float)
    result = criterion_account(y, np.column_stack([2 * y, -y]), y)
    np.testing.assert_allclose(result["contributions"], [2, -1])
    assert result["closure_error"] < 1e-14


def test_association_survives_common_rescaling_and_retains_prediction_error():
    rng = np.random.default_rng(31)
    c, predicted, y = rng.normal(size=(3, 30))
    parts = np.column_stack([predicted, c - predicted])
    a = criterion_account(c, parts, y)
    b = criterion_account(4 * c, 4 * parts, y)
    np.testing.assert_allclose(a["contributions"], b["contributions"])
    assert a["closure_error"] < 1e-14
    with pytest.raises(ValueError):
        criterion_account(c, parts + 1, y)
    with pytest.raises(ValueError):
        criterion_account(c, parts, np.ones(30))
