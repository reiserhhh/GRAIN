#!/usr/bin/env python3
"""Material-order control keeps every held author's views outside fitting."""

import numpy as np
import pytest

from scripts.analyze_grain_material_first_baseline import material_predictions


def test_no_view_from_held_author_enters_fitting():
    rng = np.random.default_rng(67)
    x = rng.normal(size=(12, 3, 7))
    y = x[:, :, 0] + rng.normal(size=(12, 3))
    folds = np.arange(12) % 4
    first = material_predictions(x, y, folds)
    y[folds == 0] += 100
    second = material_predictions(x, y, folds)
    assert first[folds == 0] == pytest.approx(second[folds == 0])


def test_view_permutation_changes_neither_mean_nor_author_folds():
    rng = np.random.default_rng(68)
    x = rng.normal(size=(12, 3, 4))
    y = x[:, :, 0]
    folds = np.arange(12) % 4
    first = material_predictions(x, y, folds)
    second = material_predictions(x[:, ::-1], y[:, ::-1], folds)
    assert first.mean(1) == pytest.approx(second.mean(1))
