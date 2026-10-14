#!/usr/bin/env python3
"""Algebraic controls, not evidence of human personality validity."""

import numpy as np
import pytest
from scipy.special import log_softmax, logsumexp

from grain_research.construction.math_context_diagnostics import (
    aligned_states,
    cross_fields,
    field_account,
)


def from_logits(z, token=0):
    """Build a four-corner field independently of the hidden-state construction."""
    z = np.asarray(z, float)
    u = z[0] - z[1] - z[2] + z[3]
    a = logsumexp(z, -1)
    normalizer = a[0] - a[1] - a[2] + a[3]
    result = field_account(u, (z[2] + z[3]) / 2, normalizer, [token])
    p = log_softmax(z, -1)
    k = p[0] - p[1] - p[2] + p[3]
    return result, k, u


def test_common_tilt_changes_offset_not_relative_diagnostics():
    hr, lr = np.array([2.0, 0, -1]), np.array([0.0, 2, -1])
    shift = np.array([1.0, 0, 0])
    result, k, _ = from_logits([[hr + shift], [lr + shift], [hr], [lr]])
    assert result["oscillation"][0] == 0
    assert result["weighted_std"][0] == 0
    assert abs(result["calibration"][0]) > 0.1
    np.testing.assert_allclose(k, k[0, 0], atol=1e-14)


def test_target_specific_tilt_changes_relative_diagnostics():
    z = np.array([[[2, -1, 1]], [[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]])
    result, k, u = from_logits(z)
    assert result["oscillation"][0] == 3
    np.testing.assert_allclose(k[:, 0] - k[:, 1], u[:, 0] - u[:, 1])


def test_gauge_cancels_in_shape_offset_and_cross_ratio():
    z = np.arange(12, dtype=float).reshape(4, 1, 3) ** 2 / 17
    first, _, _ = from_logits(z)
    second, _, _ = from_logits(z + np.array([7, -4, 11, 3])[:, None, None])
    for key in ("selected_shape", "calibration", "reconstructed_k", "oscillation"):
        np.testing.assert_allclose(first[key], second[key], atol=1e-13)


def test_invertible_dual_coordinates_preserve_field():
    rng = np.random.default_rng(3)
    h, w = rng.normal(size=(4, 1, 3)), rng.normal(size=(7, 3))
    b = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    first, _, _ = from_logits(h @ w.T)
    second, _, _ = from_logits((h @ b) @ (w @ np.linalg.inv(b).T).T)
    for key in ("selected_shape", "calibration", "oscillation", "weighted_std"):
        np.testing.assert_allclose(first[key], second[key], atol=1e-13)


def test_centered_decomposition_closes_observed_log_ratio():
    z = np.arange(12, dtype=float).reshape(4, 1, 3) ** 2 / 17
    result, k, _ = from_logits(z, token=2)
    np.testing.assert_allclose(
        result["selected_shape"] + result["calibration"], k[:, 2], atol=1e-14
    )


def test_probability_perturbation_preserves_pair_diagnostics():
    p = np.array([[0.2, 0.3, 0.5], [0.5, 0.2, 0.3]])
    q = np.array([0.8, 0.1, 0.1])
    tilted = p * q
    tilted /= tilted.sum(1)[:, None]
    ratio = np.log(p[0] / p[1])
    new_ratio = np.log(tilted[0] / tilted[1])
    np.testing.assert_allclose(ratio - ratio[0], new_ratio - new_ratio[0])


def test_shared_specific_energy_identity_without_clipping():
    result = cross_fields([[1, 0, -1]], [[-2, 0, 2]], [[0, 0, 0]])
    assert result["cosine"][0] == pytest.approx(-1)
    np.testing.assert_allclose(
        (result["variance_E"] + result["variance_N"]) / 2,
        result["shared_variance"] + result["specific_variance"],
    )
    zero = cross_fields([[0, 0, 0]], [[1, 0, -1]], [[0, 0, 0]])
    assert np.isnan(zero["cosine"][0])


def test_missing_nonfinite_and_invalid_ids_rejected():
    with pytest.raises(ValueError):
        aligned_states(np.zeros((4, 16, 1536)), np.zeros((4, 16)))
    with pytest.raises(ValueError):
        field_account([[1, np.nan]], [[0, 0]], [0], [0])
    with pytest.raises(ValueError):
        field_account([[1, 2]], [[0, 0]], [0], [2])
