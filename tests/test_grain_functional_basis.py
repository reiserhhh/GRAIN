#!/usr/bin/env python3
"""Synthetic checks for leakage boundaries and supervised functional coordinates."""

import numpy as np
import pytest

from scripts.analyze_grain_functional_basis import (
    author_folds,
    fit_estimator,
    paired_interval,
    predict,
    response_basis,
)


def test_all_repeated_author_rows_share_fold():
    groups = np.repeat([f"a{k}" for k in range(15)], 8)
    folds = author_folds(groups)
    assert len(set(folds)) == 5
    assert all(len(set(folds[groups == name])) == 1 for name in set(groups))


def test_group_cv_rejects_too_few_authors():
    with pytest.raises(ValueError):
        author_folds(["a"] * 10)


def test_pls_can_keep_low_variance_high_function_direction():
    x = np.array([[100, 1], [100, -1], [-100, 1], [-100, -1]], float)
    y = x[:, 1]
    svd = fit_estimator(x, y, response_basis(x, y, "svd_ridge", 1), 1.0)
    pls = fit_estimator(x, y, response_basis(x, y, "pls_ridge", 1), 1.0)
    assert np.mean((predict(pls, x) - y) ** 2) < 0.05
    assert np.mean((predict(svd, x) - y) ** 2) > 0.9


@pytest.mark.parametrize("method", ["svd_ridge", "full_ridge", "pls_ridge"])
def test_frozen_models_preserve_origin_and_batch_independence(method):
    rng = np.random.default_rng(10)
    x, y = rng.normal(size=(30, 40)), rng.normal(size=30)
    model = fit_estimator(x, y, response_basis(x, y, method, 4), 10.0)
    assert predict(model, np.zeros((1, 40)))[0] == 0.0
    assert np.isclose(predict(model, x[:1])[0], predict(model, x)[0])


def test_bootstrap_uses_author_mean_errors():
    y, p, baseline = np.zeros(8), np.ones(8), np.zeros(8)
    result = paired_interval(y, p, baseline, np.repeat(["a", "b"], 4), 100)
    assert result["mse_delta_vs_svd"] == 1.0
    assert result["mse_delta_lower"] == 1.0
