#!/usr/bin/env python3
"""Absolute observation offset is different from relative author distortion."""

import numpy as np
import pytest

from scripts.analyze_grain_context_reference_consequences import SUBSETS, contrast


def test_common_error_cancels_all_relative_profiles():
    coords = np.array([(x, y, g) for x in (-1, 1) for y in (-1, 1) for g in (-1, 1)])
    common = np.tile([31.0, -17.0], (8, 1))
    for _, subset in SUBSETS:
        np.testing.assert_allclose(contrast(coords, common, subset), 0, atol=1e-12)


def test_context_error_distorts_choice_not_ownership_character():
    coords = np.array([(x, y, g) for x in (-1, 1) for y in (-1, 1) for g in (-1, 1)])
    error = np.column_stack((coords[:, 0], 2 * coords[:, 1]))
    for name, subset in SUBSETS:
        value = contrast(coords, error, subset)
        if name in ("x", "y"):
            assert np.linalg.norm(value) > 0.9
        else:
            np.testing.assert_allclose(value, 0, atol=1e-12)


def test_invalid_profile_support_not_filled():
    with pytest.raises(ValueError, match="Complete"):
        contrast(np.ones((8, 3)), np.ones((8, 2)), (0,))
