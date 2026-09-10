#!/usr/bin/env python3
"""Verify complete/local distinctions and multi-condition error decomposition."""

import numpy as np

from scripts.analyze_grain_query_pair import scalar_metrics, variation_account


def test_offsets_are_separate_per_condition_and_candidate():
    rng = np.random.default_rng(19)
    y = rng.normal(size=(20, 2, 10))
    p = y + np.arange(20).reshape(2, 10)
    a = variation_account(y, p)
    assert a["variation_mse"] < 1e-26
    np.testing.assert_allclose(a["offset_mse"], np.mean(np.arange(20) ** 2))
    assert abs(a["mse"] - a["offset_mse"] - a["variation_mse"]) < 1e-12


def test_perfect_baseline_can_hide_wrong_local_action():
    rng = np.random.default_rng(22)
    base = rng.normal(size=100)
    true_action = rng.normal(size=100) * 0.01
    wrong_action = -true_action
    whole = np.corrcoef(base + true_action, base + wrong_action)[0, 1]
    assert whole > 0.99
    assert np.corrcoef(true_action, wrong_action)[0, 1] < -0.999


def test_scalar_bootstrap_keeps_paired_conditions_on_author_axis():
    y = np.arange(12, dtype=float).reshape(6, 2)
    p = y + 2
    draws = [np.array([0, 0, 1, 3, 4, 5]), np.array([2, 2, 3, 3, 4, 5])]
    result = scalar_metrics(y, p, draws)
    assert result["bootstrap_n"] == 2
    assert result["rmse"] == 2
    assert result["r_low"] > 0.999


def test_constant_prediction_has_no_author_variation():
    y = np.arange(100, dtype=float).reshape(20, 5)
    a = variation_account(y, np.full_like(y, 0.7))
    assert a["variation_mse_over_target"] == 1
    assert np.isnan(a["author_centered_r"])
