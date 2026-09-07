"""Exact squared-error attribution is not independently causal attribution."""

import numpy as np
import pytest

from scripts.analyze_grain_operator_error import decompose


def test_compensating_errors_do_not_mean_better_gap():
    result = decompose(np.array([1.0, 2.0]), np.array([2.0, 4.0]), np.array([1.0, 2.0]))
    assert result["total_error_energy"] == 0
    assert result["amplitude_energy"] == result["shape_energy"] == 1
    assert result["cross_energy"] == -2
    assert result["identity_error"] == 0


def test_undefined_denominator_is_not_an_improvement():
    with pytest.raises(ValueError):
        decompose(np.zeros(3), np.ones(3), np.ones(3))
