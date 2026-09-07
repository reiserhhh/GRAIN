#!/usr/bin/env python3
"""Target recovery checks do not infer success from score correlation alone."""

import numpy as np
import pytest

from scripts.analyze_grain_query_recovery import mse_gain


def test_exact_target_recovers_null_error_and_matched_control_zero():
    y = np.arange(1, 12, dtype=float)
    zero = np.zeros_like(y)
    result = mse_gain(y, y, zero, zero, draws=50)
    assert result["mse_gain_over_null"] == pytest.approx(1)
    assert result["mse_gain_lower"] == pytest.approx(1)
    same = mse_gain(y, y, y, zero, draws=50)
    assert same["mse_gain_upper"] == 0


def test_overscaling_can_be_harmful_despite_perfect_correlation():
    y = np.arange(-5, 6, dtype=float)
    result = mse_gain(y, 10 * y, np.zeros_like(y), np.zeros_like(y), draws=50)
    assert result["mse_gain_upper"] < 0
    with pytest.raises(ValueError):
        mse_gain(y, y[:2], y, y)
