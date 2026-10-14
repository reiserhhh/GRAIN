#!/usr/bin/env python3
"""Synthetic tests for terminal-only, complete-support numerical readback."""

import numpy as np
import pytest

from grain_research.construction.math_conditional_chart import decompose
from scripts.readback_grain_actor_action_hm116 import characters, complete_groups


def test_all_characters_match_separate_matrix_implementation():
    """Check all signed coordinates, not only the desired actor interaction."""
    x = np.random.default_rng(116).normal(size=(2, 2, 2, 2, 6, 4))
    np.testing.assert_allclose(
        characters(x), decompose(x.reshape(8, -1)).reshape(8, 2, 6, 4), atol=1e-14
    )


def test_partial_group_is_not_filled_or_promoted():
    """One missing cell makes its entire eight-cell group unavailable."""
    x = np.full((5, 2, 6, 2, 2, 2, 2, 6, 4), np.nan)
    x[0, 0, 0] = 0
    x[1, 0, 0] = 1
    x[1, 0, 0, 0, 0, 0] = np.nan
    assert complete_groups(x) == [(0, 0, 0)]


def test_unknown_support_rejected():
    """Unknown evidence cannot become a zero effect by integration."""
    with pytest.raises(ValueError):
        characters(np.full((2, 2, 2, 4), np.nan))
