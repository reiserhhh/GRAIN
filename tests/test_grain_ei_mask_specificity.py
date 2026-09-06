#!/usr/bin/env python3
"""Cross-mask paired effects must not confuse within-family and between-family axes."""

import numpy as np
import pytest

from scripts.analyze_grain_ei_mask_specificity import analyze, contrasts


def test_contrast_axes_are_estimand_by_quantity():
    values = np.arange(3 * 2 * 6).reshape(3, 2, 6)
    result = contrasts(values)
    assert result.shape == (3, 3, 2)
    for i, (a, b) in enumerate(((1, 0), (2, 0), (1, 2))):
        for j, q in enumerate((0, 4, 5)):
            for k in range(2):
                assert result[i, j, k] == values[a, k, q] - values[b, k, q]


def test_identical_arms_have_exact_zero_paired_effect_and_interval():
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 100)
    folds = np.repeat(np.arange(5), 40)
    scores = rng.normal(size=(7, 200)) + 0.2 * y
    rows = analyze(y, np.stack([scores] * 3), folds, draws=24)
    assert len(rows) == 18
    assert all(r["estimate"] == r["lower"] == r["upper"] == 0 for r in rows)
    with pytest.raises(ValueError):
        analyze(y, np.stack([scores] * 2), folds, draws=24)
