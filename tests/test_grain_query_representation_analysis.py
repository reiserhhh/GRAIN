#!/usr/bin/env python3
"""Quality metrics distinguish target fit from calibration and sample grouping."""

import numpy as np
import pytest

from scripts.analyze_grain_functional_basis import author_folds
from scripts.analyze_grain_query_representation import target_quality


def test_target_fidelity_is_separate_from_mse_and_no_test_centering():
    y = np.array([-2.0, -1.0, 1.0, 2.0])
    exact = target_quality(y, y, np.full(4, 0.2))
    assert exact["teacher_r"] == pytest.approx(1)
    assert exact["teacher_mse_over_null"] == 0
    inflated = target_quality(y, 10 * y, np.full(4, 0.2))
    assert inflated["teacher_r"] == pytest.approx(1)
    assert inflated["teacher_mse_over_null"] > 1
    with pytest.raises(ValueError):
        target_quality(y, y[:3], np.zeros(4))


def test_source_fold_assignment_does_not_depend_on_array_order_or_repeated_views():
    names = np.array(["u" + str(i) for i in range(25)])
    folds = author_folds(names, 5, "s4_outer")
    reverse = author_folds(names[::-1], 5, "s4_outer")
    np.testing.assert_array_equal(folds, reverse[::-1])
    repeats = author_folds(np.repeat(names, 3), 5, "s4_outer")
    np.testing.assert_array_equal(repeats, np.repeat(folds, 3))
