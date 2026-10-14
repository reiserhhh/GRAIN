#!/usr/bin/env python3
"""Tests of HM74 consequence units, preserved interactions and profile controls."""

import json

import numpy as np
import pytest

from scripts.interpret_grain_causal_ownership_sectors import (
    SIGNS,
    actions,
    check_profiles,
    coefficient_table,
    scalar_account,
)


def test_conditional_action_and_marginal_cancellation():
    """Retain context-dependent actions even if the average cancels."""
    assert np.array_equal(actions(SIGNS[7]), [0, -4, 4])
    assert np.array_equal(actions(SIGNS[3]), [4, 4, 4])


def test_coefficients_all_sectors_and_reference_retained():
    """Known intercept and every interaction replay without selecting a rank."""
    coef = np.arange(8, dtype=float)
    assert np.array_equal(coefficient_table(SIGNS.T @ coef), coef)


def test_zero_and_reversed_effect_not_conflated_with_reduction():
    """Signed reversal has unchanged magnitude; absent evidence has no ratio."""
    reverse = scalar_account([1, -1], [-1, 1])
    assert reverse["rms_reduction_fraction"] == 0
    assert reverse["change_rms"] == 2
    absent = scalar_account([0, 0], [1, 1])
    assert absent["remaining_rms_ratio"] is None
    assert absent["rms_reduction_fraction"] is None


@pytest.mark.parametrize("j", [3, 7])
def test_full_probability_and_consequence_audit(j):
    """All declared lower marginals and score contrasts have literal checks."""
    value = SIGNS[3] + 0.25 * SIGNS[7]
    plus, minus = (1 + 0.5 * SIGNS[j]) / 8, (1 - 0.5 * SIGNS[j]) / 8
    row = {
        "character": "yg" if j == 3 else "xyg",
        "plus_probability": json.dumps(plus.tolist()),
        "minus_probability": json.dumps(minus.tolist()),
        "plus_score": json.dumps(float(plus @ value)),
        "minus_score": json.dumps(float(minus @ value)),
        "difference": json.dumps(float((plus - minus) @ value)),
    }
    assert check_profiles(row, value) < 1e-12
    row["plus_probability"] = json.dumps(minus.tolist())
    with pytest.raises(ValueError, match="Undeclared"):
        check_profiles(row, value)


def test_incomplete_coefficient_input_rejected():
    """Missing cells cannot masquerade as a zero interaction."""
    with pytest.raises(ValueError):
        coefficient_table(np.zeros(7))
