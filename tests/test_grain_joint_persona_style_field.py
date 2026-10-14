#!/usr/bin/env python3
"""Signed field diagnostics distinguish direction, moderation and stochasticity."""

import numpy as np
import pytest

from scripts.analyze_grain_joint_persona_style_field_hm150 import (
    crossed_product,
    response_account,
)


def test_unchanged_direction_has_zero_mixed_support():
    x = np.array([[1.0, 2.0]] * 3)
    result = response_account(x, x)
    assert result["mixed_support_per_coordinate"] == 0
    assert result["mean_direction_cosine"] == pytest.approx(1)


def test_rotation_can_modulate_without_negative_alignment():
    x = np.array([[1.0, 0.0]] * 3)
    y = np.array([[1.0, 1.0]] * 3)
    result = response_account(x, y)
    assert result["mixed_support_per_coordinate"] > 0
    assert result["alignment_per_coordinate"] > 0


def test_reversal_has_supported_directions_and_negative_alignment():
    x = np.array([[1.0, 2.0]] * 3)
    result = response_account(x, -x)
    assert result["neutral_support_per_coordinate"] > 0
    assert result["styled_support_per_coordinate"] > 0
    assert result["alignment_per_coordinate"] < 0
    assert result["mean_direction_cosine"] == pytest.approx(-1)


def test_seed_fluctuation_not_clipped_to_positive_energy():
    x = np.array([[1.0], [-1.0], [0.0]])
    assert crossed_product(x, x) == pytest.approx(-1 / 3)


def test_zero_mean_cosine_not_manufactured():
    result = response_account(np.zeros((3, 2)), np.ones((3, 2)))
    assert result["mean_direction_cosine"] is None


def test_incompatible_units_fail():
    with pytest.raises(ValueError, match="aligned"):
        crossed_product(np.zeros((3, 2)), np.zeros((3, 3)))
