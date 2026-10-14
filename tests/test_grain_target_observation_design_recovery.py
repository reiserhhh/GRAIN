#!/usr/bin/env python3
"""The compatibility guard must not rescue material errors or change ordinary laws."""

import numpy as np
import pytest

from grain_research.construction import math_target_observation_design as M
from scripts.run_grain_target_observation_design_hm206_recovery import (
    entropy_compatibility,
)


def test_ulp_complement_is_recorded_without_changing_positive_entry():
    p = np.array([1 + np.finfo(float).eps, -np.finfo(float).eps])
    original = M.entropy
    with entropy_compatibility() as receipt:
        result = M.entropy(p)
        assert result == original(np.array([p[0], 0.0]))
        assert receipt["repaired_entries"] == 1
        assert receipt["minimum_entry"] == p[1]
    assert M.entropy is original


def test_large_negative_and_nonfinite_still_fail():
    with entropy_compatibility():
        for p in ([1.0001, -0.0001], [np.nan, 0.5]):
            with pytest.raises(ValueError):
                M.entropy(p)


def test_ordinary_entropy_and_source_scores_are_unchanged():
    p = np.zeros((4, 2, 8))
    p[:, 0, 0] = 1
    p[:2, 1, 0], p[2:, 1, 1] = 1, 1
    y = np.repeat(np.array([0, 0, 1, 1])[:, None], 5, axis=1)
    original = M.compile_design(p, y)
    with entropy_compatibility() as receipt:
        recovered = M.compile_design(p, y)
    assert receipt["repaired_entries"] == 0
    for key in original:
        np.testing.assert_allclose(original[key], recovered[key], equal_nan=True)
