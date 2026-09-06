#!/usr/bin/env python3
"""AUC arithmetic and fixed-fold bootstrap instrumentation on synthetic data."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts.analyze_grain_ei_source_gap import analyze, gap_values, rank_auc


def test_rank_auc_ties_and_invalid_support():
    y = np.array([0, 1, 0, 1, 1, 0])
    for p in ([0, 0, 1, 1, 1, 0], [1] * 6, np.arange(6)):
        assert rank_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert np.isnan(rank_auc(np.zeros(6), np.ones(6)))
    with pytest.raises(ValueError):
        rank_auc(y, [float("nan")] * 6)


def test_decomposition_averages_scores_not_predictions():
    rng = np.random.default_rng(17)
    y = np.arange(100) % 2
    p = rng.normal(size=(7, 100))
    v = gap_values(y, p)
    assert v[2] == pytest.approx(np.mean([roc_auc_score(y, q) for q in p[2:]]))
    assert v[3] + v[4] == pytest.approx(v[5], abs=1e-14)
    with pytest.raises(ValueError, match="five"):
        gap_values(y, p[:6])


def test_pooling_can_change_without_any_within_fold_rank_change():
    n = 200
    folds = np.arange(n) % 5
    y = ((np.arange(n) // 5) % 2).astype(float)
    p = y + np.arange(n) / 1000
    shifted = p + 10 * folds
    assert rank_auc(y, p) > rank_auc(y, shifted)
    for f in range(5):
        ix = folds == f
        assert rank_auc(y[ix], p[ix]) == rank_auc(y[ix], shifted[ix])
    predictions = np.stack([p, shifted, *[p] * 5])
    rows, per_fold = analyze(y, predictions, folds, draws=20)
    target = next(
        r
        for r in rows
        if r["estimand"] == "mean_official_fold"
        and r["quantity"] == "purge_minus_count"
    )
    assert target["estimate"] == 0 and target["lower"] == 0 and target["upper"] == 0
    assert np.max(abs(per_fold[:, 3] + per_fold[:, 4] - per_fold[:, 5])) < 1e-14
