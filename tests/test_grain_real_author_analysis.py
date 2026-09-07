#!/usr/bin/env python3
"""Small synthetic checks for the fixed D2-D4 mathematical summaries."""

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_grain_real_author_path import (
    cluster_mean,
    finite_curvature,
    fit_low_rank,
    prediction_quality,
)


def test_author_is_bootstrap_unit():
    frame = pd.DataFrame(
        {"author_id": ["a", "a", "b", "b"], "effect": [0.0, 2.0, 3.0, 5.0]}
    )
    result = cluster_mean(frame, "effect", draws=100)
    assert result["authors"] == 2
    assert result["mean"] == 2.5


def test_compact_operator_recovers_low_rank_response():
    rng = np.random.default_rng(42)
    axis = rng.normal(size=(20, 2))
    train = rng.normal(size=(80, 2)) @ axis.T
    held = rng.normal(size=(40, 2)) @ axis.T
    truth = rng.normal(size=20)
    basis, weights = fit_low_rank(train, train @ truth, rank=2, ridge=1e-8)
    pred = held @ basis @ weights
    assert prediction_quality(held @ truth, pred)["r"] == pytest.approx(1.0)
    assert np.zeros(20) @ basis @ weights == 0


def test_training_basis_orthonormal():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(10, 30))
    basis, weights = fit_low_rank(x, rng.normal(size=10), 8)
    assert basis.shape == (30, 8)
    assert weights.shape == (8,)
    assert np.allclose(basis.T @ basis, np.eye(8))


def test_zero_response_does_not_divide_by_zero():
    assert prediction_quality(np.zeros(3), np.zeros(3))["mse_ratio_to_zero"] is None


def test_finite_curvature_separates_linear_and_quadratic_response():
    assert finite_curvature(0.2, 0.3, 0.4) == pytest.approx(0.0)
    # For f(alpha)=.2+.1*alpha+.4*alpha**2, the contrast equals .4/2.
    assert finite_curvature(0.2, 0.35, 0.7) == pytest.approx(0.2)
