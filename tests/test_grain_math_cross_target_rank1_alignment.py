#!/usr/bin/env python3
"""Rank-one label alignment must not be confused with multiple selective modes."""

import numpy as np
import pytest

from scripts.analyze_grain_math_cross_target_rank1_alignment import (
    diagnostics,
    leading_component,
)


def test_rank1_can_have_unique_named_matching():
    """A good label matching alone does not refute a single generic mode."""
    d = np.outer(np.arange(1, 7), np.arange(1, 7))
    first, remainder, _ = leading_component(d)
    assert diagnostics(d)["named_rank"] == 1
    assert diagnostics(d)["tied_assignments"] == 1
    np.testing.assert_allclose(d, first, atol=1e-13)
    np.testing.assert_allclose(remainder, 0, atol=1e-13)


def test_genuine_target_specific_tail_is_retained():
    """A dominant common mode does not erase a selective tail from the report."""
    d = np.ones((6, 6)) * 10 + np.eye(6)
    first, remainder, _ = leading_component(d)
    assert diagnostics(first)["tied_assignments"] == 720
    assert diagnostics(remainder)["named_rank"] == 1
    assert diagnostics(remainder)["tied_assignments"] == 1
    assert diagnostics(remainder)["named_diagonal_excess"] == pytest.approx(1)


def test_equal_leading_modes_are_not_arbitrarily_named():
    """Degenerate leading vectors have no unique label alignment."""
    first, remainder, gap = leading_component(np.eye(6))
    assert first is None and remainder is None and gap == 0


def test_zero_operator_does_not_get_a_leading_axis():
    """All-zero response must not create a perfect residual label fit."""
    assert leading_component(np.zeros((6, 6))) == (None, None, 0)


def test_components_preserve_named_excess_and_energy():
    """Component alignment can be recombined exactly without a predictive fit."""
    d = np.arange(36).reshape(6, 6) / 7 + np.eye(6) * 0.3
    first, remainder, _ = leading_component(d)
    assert abs(np.sum(first * remainder)) < 1e-12
    x = diagnostics(d)["named_diagonal_excess"]
    assert x == pytest.approx(
        diagnostics(first)["named_diagonal_excess"]
        + diagnostics(remainder)["named_diagonal_excess"]
    )
    np.testing.assert_allclose(d, first + remainder, atol=1e-14)
