#!/usr/bin/env python3
"""Isometry, observation-completion uniqueness, and nonisometric controls."""

import numpy as np
import pytest

from grain_research.construction import math_relational_isometry as M
from grain_research.construction.math_relational_dual_frame import fit_frame


def fixture():
    """Build a synthetic rank-six relation support and full orthogonal change."""
    rng = np.random.default_rng(126)
    x = rng.normal(size=(30, 6))
    rotation = np.linalg.qr(rng.normal(size=(30, 30)))[0]
    polar = M.supported_polar(x, rotation @ x)
    frame = fit_frame(np.zeros((1, 30)), x)
    bundle = {
        "source_center": frame["center"],
        "context_center": frame["center"],
        "source_columns": x,
        "dual": frame["dual"],
    }
    for prefix in ("matched", "wrong"):
        bundle.update(
            {prefix + "_left": polar["left"], prefix + "_right": polar["right"]}
        )
    return rng, x, rotation, bundle


def test_exact_rotation_transports_numeric_coordinates():
    rng, x, rotation, bundle = fixture()
    theta = rng.normal(size=(10, 6))
    noise = rng.normal(size=(10, 30))
    noise -= noise @ x @ np.linalg.pinv(x)
    values = (theta @ x.T + noise) @ rotation.T
    scored = M.score_vectors(values, bundle, "matched_anchors")
    np.testing.assert_allclose(scored["coordinates"], theta, atol=1e-12)
    np.testing.assert_allclose(
        scored["complement"] @ bundle["matched_left"], 0, atol=1e-12
    )


def test_best_rotation_formula_agrees_with_explicit_fit():
    rng, x, _, _ = fixture()
    account = M.gram_account(x, rng.normal(size=x.shape))
    assert (
        abs(
            account["best_rotation_residual_sq"]
            - account["direct_rotation_residual_sq"]
        )
        < 1e-10
    )


def test_unobserved_rotation_completion_does_not_change_supported_score():
    rng, x, rotation, bundle = fixture()
    v = rng.normal(size=30)
    v -= x @ np.linalg.lstsq(x, v, rcond=None)[0]
    v /= np.linalg.norm(v)
    other = rotation @ (np.eye(30) - 2 * np.outer(v, v))
    np.testing.assert_allclose(rotation @ x, other @ x, atol=1e-12)
    y = rng.normal(size=30)
    a, b = bundle["dual"] @ rotation.T @ y, bundle["dual"] @ other.T @ y
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert np.linalg.norm(rotation.T @ y - other.T @ y) > 1e-3


def test_shape_change_cannot_be_called_pure_rotation():
    _, x, _, _ = fixture()
    account = M.gram_account(x, x * np.arange(1, 7))
    assert account["relative_best_rotation_residual"] > 0.5
    assert account["normalized_gram_shape_gap"] > 0.1


def test_gram_does_not_identify_semantic_polarity():
    _, x, _, _ = fixture()
    account = M.gram_account(x, -x)
    np.testing.assert_allclose(account["source_gram"], account["target_gram"])
    assert account["normalized_gram_shape_gap"] == 0


def test_rank_failure_is_not_filled_with_scores():
    with pytest.raises(ValueError, match="Rank-deficient"):
        M.supported_polar(np.ones((8, 3)), np.ones((8, 3)))
