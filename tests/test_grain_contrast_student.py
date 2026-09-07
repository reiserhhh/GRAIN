#!/usr/bin/env python3
"""Checks that decomposition alone does not improve a fixed linear student."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts.run_grain_contrast_student import common_operator_error, select, specification


def test_same_pca_ridge_operator_is_additive_in_target():
    rng = np.random.default_rng(11)
    x, xt = rng.normal(size=(25, 12)), rng.normal(size=(7, 12))
    a, b = rng.normal(size=25), rng.normal(size=25)
    with threadpool_limits(limits=2):
        assert common_operator_error(x, a, b, xt) < 1e-12


def test_fixed_family_and_inner_selection_need_no_questionnaire():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(24, 10))
    teacher = x[:, 0] + 0.2 * rng.normal(size=24)
    with threadpool_limits(limits=2):
        alpha = select(x, teacher, np.array([str(i) for i in range(24)]), "pls2", "unit")
    assert alpha in (1.0, 10.0, 100.0)
    with pytest.raises(ValueError):
        specification("outcome_selected_family")
