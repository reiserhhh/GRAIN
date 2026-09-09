#!/usr/bin/env python3
"""Synthetic checks for actual-factor/path joins, fit boundaries and directions."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.factor_path_routes import (
    aligned_directions,
    design_matrix,
    fold_diagnostics,
    historical_overlap,
    metrics,
    nested_predictions,
    paired_delta,
    split_direction,
)


def test_design_keeps_independent_coordinates():
    g, p = np.ones((8, 12)), np.arange(24).reshape(8, 3)
    assert design_matrix(g, p, "grain").shape == (8, 12)
    assert design_matrix(g, p, "path").shape == (8, 3)
    assert design_matrix(g, p, "joint").shape == (8, 15)
    assert design_matrix(g, p, "interaction").shape == (8, 51)
    with pytest.raises(ValueError):
        design_matrix(g, p, "not_an_arm")


def test_no_repeated_author_rows():
    with pytest.raises(ValueError, match="one row per author"):
        nested_predictions(np.ones((20, 2)), np.arange(20), ["a"] * 20)


def test_outer_labels_cannot_change_own_predictions():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, 3))
    y = x[:, 0] + rng.normal(size=40) * 0.1
    authors = np.arange(40)
    a, folds, _ = nested_predictions(x, y, authors)
    edited = y.copy()
    edited[folds == 0] += 10000
    b, folds2, _ = nested_predictions(x, edited, authors)
    np.testing.assert_array_equal(folds, folds2)
    np.testing.assert_allclose(a[folds == 0], b[folds == 0])


def test_direction_identity_null_norm_and_split():
    rng = np.random.default_rng(4)
    g = rng.normal(size=(100, 4))
    truth = rng.normal(size=(4, 18))
    h = g @ truth
    real, null, center, scale = aligned_directions(g, h, alpha=0.0001)
    np.testing.assert_allclose(real / scale[:, None], truth, atol=1e-4)
    np.testing.assert_allclose(
        np.linalg.norm(real, axis=1), np.linalg.norm(null, axis=1)
    )
    assert center.shape == scale.shape == (4,)
    p, q = split_direction(real[0], (2, 6), (6,))
    np.testing.assert_array_equal(np.r_[p.ravel(), q], real[0])
    with pytest.raises(ValueError):
        split_direction(real[0], (4, 6), (6,))


def test_paired_bootstrap_identical_predictions():
    y = np.arange(30) % 2
    p = np.arange(30) * 0.01 + y
    d = paired_delta(y, p, p, binary=True, draws=40)
    for key in (
        "delta_r",
        "delta_r_lower",
        "delta_r_upper",
        "delta_auc",
        "delta_auc_lower",
        "delta_auc_upper",
    ):
        assert d[key] == 0
    assert metrics(y, p, binary=True)["auc"] == 1


def test_nonfinite_and_constant_directions_rejected():
    with pytest.raises(ValueError):
        aligned_directions(np.ones((20, 4)), np.ones((20, 8)))
    with pytest.raises(ValueError):
        design_matrix(np.full((2, 4), np.nan), np.ones((2, 3)), "joint")


def test_raw_and_pseudonym_identity_comparison():
    from grain_research.vector_mechanisms.data_views import stable_key

    target = stable_key("author_a", "test_seed")[:24]
    assert historical_overlap(["author_a", "author_b"], [target], seed="test_seed") == {
        target
    }
    assert historical_overlap(["author_a"], [target], seed="different_seed") == set()


def test_cv_null_exposes_pooled_intercept_bias():
    y = np.r_[np.arange(10), np.arange(10) + 5, np.arange(10) + 10]
    folds = np.repeat(np.arange(3), 10)
    null = np.array([y[folds != f].mean() for f in folds])
    d = fold_diagnostics(y, null, folds)
    assert d["training_mean_pooled_r"] < 0
    assert d["mse_gain_vs_training_mean"] == 0
    assert d["within_fold_centered_r"] is None
