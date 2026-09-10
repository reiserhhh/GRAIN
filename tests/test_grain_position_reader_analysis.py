#!/usr/bin/env python3
"""Correlation must not conceal a scale error or manufacture constant fidelity."""

import numpy as np
import pytest

from scripts.analyze_grain_position_reader import action_metrics


def test_correct_sign_and_perfect_r_can_still_have_large_action_error():
    y = np.array([-2.0, -1.0, 1.0, 2.0])
    p = 3 * y
    m = action_metrics(y, p)
    assert m["action_r"] > 0.999
    assert m["action_mse_over_zero"] == 4
    assert m["same_sign"] == 4


def test_zero_reader_has_reference_error_one_and_undefined_r():
    m = action_metrics(np.arange(1.0, 5.0), np.zeros(4))
    assert m["action_mse_over_zero"] == 1
    assert np.isnan(m["action_r"])
    assert m["same_sign"] == 0


def test_bad_tangent_shapes_are_rejected():
    with pytest.raises(ValueError):
        action_metrics(np.ones((4, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError):
        action_metrics(np.arange(4.0), np.ones(5))
