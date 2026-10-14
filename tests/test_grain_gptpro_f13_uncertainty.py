"""Exact joint-change and total-variance identities, including invalid inputs."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f13_uncertainty import (
    change_interval,
    mixture_components,
)


def test_impossible_covariance_not_silently_zeroed():
    with pytest.raises(ValueError):
        change_interval([0, 1], [[1, 2], [2, 1]])


def test_duplicate_measurement_has_no_independent_change_information():
    result = change_interval([2, 2], [[1, 1], [1, 1]])
    assert result["variance"] == 0 and result["change"] == 0


def test_paired_covariance_changes_mdc():
    result = change_interval([0, 1], [[1, 0.75], [0.75, 1]])
    assert result["variance"] == 0.5
    np.testing.assert_allclose(result["mdc_model"], 1.96 * np.sqrt(0.5))


def test_total_variance_matches_explicit_finite_distribution():
    means = np.array([[-2.0, 1.0], [3.0, -1.0]])
    offsets = np.array([[-1.0, -0.5], [1.0, 0.5]])
    covariance = offsets.T @ offsets / 2
    result = mixture_components(means, np.tile(covariance, (2, 1, 1)))
    all_values = np.concatenate([m + offsets for m in means])
    np.testing.assert_allclose(
        result["total"], np.cov(all_values, rowvar=False, ddof=0)
    )
    assert result["between"][0, 1] < 0


def test_single_source_fit_has_zero_between_component():
    result = mixture_components([[1, 2]], [np.eye(2)])
    np.testing.assert_array_equal(result["between"], np.zeros((2, 2)))
