#!/usr/bin/env python3
"""Known-answer tests for measurement/design references."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from grain_research.contextual_measurement import (
    basis,
    gaussian_scores,
    posterior_covariance,
    response_coordinates,
    target_variance_gain,
)


def test_response_keeps_slope_when_mean_cancels():
    z = np.tile([-1.0, 1.0], (2, 100))
    y = np.array([2.0, -3.0])[:, None] * z
    assert np.allclose(y.mean(1), 0)
    assert np.allclose(response_coordinates(z, y, penalty=1e-8)[:, 1], [2, -3])


def test_gain_is_exact_and_target_specific():
    x = basis([-0.2, 0.2])
    prior = np.array([[2.0, 0.2], [0.2, 1.0]])
    target, candidate = np.array([0.0, 1.0]), basis(0.8)
    cov = posterior_covariance(x, 1.0, prior)
    after = posterior_covariance(np.vstack([x, candidate]), [1, 1, 0.7], prior)
    assert target_variance_gain(cov, target, candidate, 0.7) == pytest.approx(
        target @ (cov - after) @ target
    )
    assert target_variance_gain(np.eye(2), target, basis(0.0), 0.1) == 0


def test_scores_independent_of_test_batch():
    x = basis([-1, 0, 1])
    y = np.array([[1, 2, 3], [-4, 10, 100]])
    full, _ = gaussian_scores(x, y, np.ones(3), np.eye(2))
    first, _ = gaussian_scores(x, y[:1], np.ones(3), np.eye(2))
    assert np.array_equal(full[:1], first)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        posterior_covariance(basis([0]), 0, np.eye(2))
    with pytest.raises(ValueError):
        response_coordinates([[0, 1]], [[0, np.nan]])


def test_frozen_world_names_are_strings():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/grain_contextual_measurement.yaml"
    )
    config = yaml.safe_load(path.read_text())
    assert len(config["worlds"]) == 8
    assert "null" in config["worlds"]
    assert all(isinstance(w, str) for w in config["worlds"])


def test_posterior_risk_calibration():
    rng = np.random.default_rng(500)
    x, noise = basis([-1, -0.5, 0.5, 1]), np.array([1, 0.5, 0.5, 1])
    beta = rng.normal(size=(30000, 2))
    y = beta @ x.T + rng.normal(size=(len(beta), 4)) * np.sqrt(noise)
    scores, cov = gaussian_scores(x, y, noise, np.eye(2))
    assert np.mean((scores[:, 1] - beta[:, 1]) ** 2) == pytest.approx(
        cov[1, 1], rel=0.04
    )
