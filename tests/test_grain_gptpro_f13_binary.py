#!/usr/bin/env python3
"""Finite probability uncertainty must not be confused with Bernoulli variance."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f13_binary import next_probability_components


def test_same_predictive_mean_different_probability_uncertainty():
    known = next_probability_components([0.5, 0.5], [0.5, 0.5])
    unknown = next_probability_components([0.5, 0.5], [0, 1])
    assert known["mean"] == unknown["mean"] == 0.5
    assert known["total_outcome_variance"] == unknown["total_outcome_variance"] == 0.25
    assert known["state_probability_variance"] == 0
    assert unknown["state_probability_variance"] == 0.25
    assert known["probability_lower"] == known["probability_upper"] == 0.5


def test_enumerated_joint_law():
    w, p = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.01, 0.3, 0.5, 0.99])
    result = next_probability_components(w, p)
    joint = w[:, None] * np.column_stack([1 - p, p])
    mean = joint[:, 1].sum()
    variance = (joint[:, 0] * mean**2 + joint[:, 1] * (1 - mean) ** 2).sum()
    np.testing.assert_allclose(result["total_outcome_variance"], variance, atol=1e-12)


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        next_probability_components([0.3, 0.3], [0.2, 0.8])
