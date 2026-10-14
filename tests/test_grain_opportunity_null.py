#!/usr/bin/env python3
"""Check mathematical degeneracy and matched-negative selection controls."""

import numpy as np

from scripts.analyze_grain_opportunity_null_hm94 import win


def test_roundoff_is_not_an_author_effect():
    np.testing.assert_allclose(win(np.array([0, 1e-20, -1e-20]), 0), 0.5)


def test_positive_and_negative_are_not_reversed():
    np.testing.assert_equal(win(np.array([2.0, 1.0, 0.0]), 1.0), [1, 0.5, 0])


def test_nearest_nuisance_strangers_can_beat_random_identity_without_trait():
    feature = np.arange(21, dtype=float)
    query = 0.0
    negative = (feature[1:6] - query) ** 2
    positive = (feature[10] - query) ** 2
    assert win(negative, positive).mean() == 0
