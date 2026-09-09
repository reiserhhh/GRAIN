#!/usr/bin/env python3
"""Loss projections, preserved geometry and non-affine error witnesses."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.reading_levels import (
    affine_error_account,
    fit_readers,
    split_loss,
    within_change,
    within_predictions,
)


def test_balanced_projection_loss_identity_with_random_offsets():
    rng = np.random.default_rng(5)
    y, p = rng.normal(size=(2, 9, 3, 2))
    result = split_loss(y, p)
    assert result["max_error"] < 1e-14
    with pytest.raises(ValueError):
        split_loss(y, p[:, :2])


def test_same_geometry_and_identified_within_changes_only():
    rng = np.random.default_rng(9)
    bags = rng.normal(size=(24, 5, 6))
    y = rng.normal(size=(24, 2))
    r = fit_readers(bags, y)
    np.testing.assert_array_equal(r["pooled"].centers[0], r["between"].centers[0])
    np.testing.assert_array_equal(r["pooled"].scales, r["between"].scales)
    p = within_predictions(r, bags).reshape(8, 3, 2)
    np.testing.assert_allclose(p.mean(1), 0, atol=1e-12)
    np.testing.assert_allclose(
        within_change(r, bags[::3], bags[1::3]), p[:, 1] - p[:, 0], atol=1e-12
    )


def test_between_and_within_can_encode_opposing_relations():
    rng = np.random.default_rng(30)
    means = rng.normal(scale=2, size=(24, 1, 1))
    noise = rng.normal(size=(24, 3, 1))
    noise -= noise.mean(1, keepdims=True)
    x = means + noise
    target = np.repeat(3 * means - 2 * noise, 2, axis=2)
    bags = np.repeat(x.reshape(-1, 1, 1), 4, axis=1)
    r = fit_readers(bags, target.reshape(-1, 2))
    assert r["between"].head.coef_[0, 0] > 0
    assert r["within_coef"][0, 0] < 0
    assert r["loss_account"]["max_error"] < 1e-12


def test_simple_shrinkage_is_not_missing_information():
    t = np.arange(20, dtype=float)
    result = affine_error_account(t, 0.1 * t + 3, t)
    assert result["alignment_slope"] == pytest.approx(10)
    assert result["amplitude_contribution"] == pytest.approx(0.9)
    assert abs(result["non_affine_contribution"]) < 1e-12


def test_nonlinear_remainder_can_retain_criterion_link_after_scale_alignment():
    p = np.linspace(-1, 1, 101)
    y = p**2
    result = affine_error_account(p + y, p, y)
    assert result["alignment_slope"] == pytest.approx(1)
    assert result["non_affine_criterion_r"] == pytest.approx(1)
    assert result["closure_error"] < 1e-12
