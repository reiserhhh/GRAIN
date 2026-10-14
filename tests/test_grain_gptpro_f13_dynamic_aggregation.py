#!/usr/bin/env python3
"""Total-covariance aggregation retains source and within-model cross terms."""

import numpy as np
import pytest

from scripts.aggregate_grain_gptpro_f13_dynamic import combine


def test_total_covariance_second_moment_identity():
    means = np.array([[1, 2], [3, -1], [0, 0.5]])
    event = np.tile([[2, 0.3], [0.3, 1]], (3, 1, 1))
    parameter = np.tile([[1, -0.2], [-0.2, 0.5]], (3, 1, 1))
    m, e, p, s, total = combine(means, event, parameter)
    expected = sum(
        event[i] + parameter[i] + np.outer(means[i], means[i]) for i in range(3)
    ) / 3 - np.outer(m, m)
    np.testing.assert_allclose(total, expected)
    np.testing.assert_allclose(total, e + p + s)
    assert s[0, 1] < 0


def test_incomplete_nonfinite_mixture_rejected():
    with pytest.raises(ValueError):
        combine([[np.nan, 0]], [np.eye(2)], [np.eye(2)])
