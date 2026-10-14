#!/usr/bin/env python3
"""Small exact observation-fiber and pullback witnesses."""

import numpy as np
import pytest
from grain_research.construction.math_decoder_fibers import (
    directional_action,
    normalize_direction,
    observation_matrix,
    split_gradient,
)


@pytest.mark.parametrize("tokens", [1, 2, 3, 4])
def test_property_moments_injective(tokens):
    """No lost temporal direction exists for these actual property lengths."""
    g = np.random.default_rng(tokens).normal(size=(tokens, 7))
    result = split_gradient(g, 3)
    assert result["rank"] == tokens
    assert np.max(abs(result["null"])) < 1e-13


def test_mean_fiber_and_pullback():
    """Moment-null response prevents factoring through the mean even locally."""
    g = np.array([[1.0, 2.0], [-1.0, 3.0], [0.0, -2.0]])
    result = split_gradient(g, 0)
    o = observation_matrix(3, 0)
    assert np.max(abs(o @ result["null"])) < 1e-13
    assert np.sum(g * result["null"]) > 1
    assert np.allclose(o.T @ result["moment_covector"], result["visible"])
    delta = np.random.default_rng(42).normal(size=g.shape)
    assert np.allclose(
        np.sum(result["visible"] * delta),
        np.sum(result["moment_covector"] * (o @ delta)),
    )


def test_joint_normalization():
    """One radius covers the entire collection, not one radius per token."""
    result = normalize_direction({"a": np.ones((3, 2)), "b": np.zeros((2, 2))}, 0.2)
    assert np.isclose(sum(np.sum(v**2) for v in result.values()), 0.04)
    assert np.all(result["b"] == 0)
    assert np.isclose(directional_action(result, result), 0.04)


def test_long_property_has_real_null():
    """The full-recovery fact must not be extrapolated to arbitrary long spans."""
    result = split_gradient(np.eye(5), 3)
    assert result["rank"] == 4
    assert result["null_tokens"] == 1
    assert result["null_observation_error"] < 1e-13


def test_invalid_data():
    """Reject invalid observations and unmatched spans rather than filling zeros."""
    with pytest.raises(ValueError):
        observation_matrix(0, 3)
    with pytest.raises(ValueError):
        split_gradient(np.array([[np.nan]]), 0)
    with pytest.raises(ValueError):
        directional_action({"a": np.ones((1, 1))}, {"b": np.ones((1, 1))})
