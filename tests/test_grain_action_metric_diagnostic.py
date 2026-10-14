#!/usr/bin/env python3
"""Synthetic tests for the bounded action-metric diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from grain_research.vector_mechanisms.action_metric_diagnostic import (
    fit_action_weighted,
    fit_euclidean_action,
    paired_metrics,
    predict_euclidean,
    predict_weighted,
    scalar_action,
)


def test_weighted_fit_uses_dual_solution_and_is_finite():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(12, 4))
    target = rng.normal(size=(12, 7))
    jac = rng.normal(size=(12, 7))
    model = fit_action_weighted(x, target, jac)
    assert model["weight"].shape == (7, 4)
    assert np.isfinite(predict_weighted(model, x)).all()
    assert model["source_loss"] >= 0


def test_euclidean_fit_and_predict():
    x = np.arange(20, dtype=float).reshape(5, 4)
    target = np.arange(35, dtype=float).reshape(5, 7)
    model = fit_euclidean_action(x, target)
    assert predict_euclidean(model, x).shape == target.shape


def test_scalar_action_and_paired_metrics():
    j = np.array([1.0, -2.0])
    assert scalar_action(j, np.array([3.0, 4.0])) == -5.0
    metrics = paired_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert metrics["r"] == pytest.approx(1.0)
    assert metrics["direction"] == 1.0


def test_rejects_unaligned_inputs():
    with pytest.raises(ValueError):
        fit_action_weighted(np.zeros((2, 4)), np.zeros((2, 3)), np.zeros((2, 2)))
