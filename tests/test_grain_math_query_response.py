#!/usr/bin/env python3
"""Finite source-only query-response tests, including identifiable failures."""

import numpy as np
import pytest

from grain_research.construction import math_query_response as R
from scripts.verify_grain_math_query_response import primal


def test_known_linear():
    d = np.array([[1, 0, 2], [0, 1, 3]], float)
    matrix = np.array([[2, 1], [-1, 4]], float)
    model = R.fit(d, matrix @ d)
    x = np.array([3, -2.0])
    np.testing.assert_allclose(R.predict(model, x)["predicted"], matrix @ x)


def test_primal_gram_equivalence():
    rng = np.random.default_rng(34)
    d, g = rng.normal(size=(30, 8)), rng.normal(size=(12, 8))
    model = R.fit(d, g)
    u, s, v, target = primal(d, g)
    x = rng.normal(size=30)
    np.testing.assert_allclose(
        R.predict(model, x)["predicted"], target @ v @ ((u.T @ x) / s), atol=1e-11
    )


def test_rank_deficient():
    d = np.array([[1, 2, 3], [0, 0, 0]], float)
    g = np.array([[2, 4, 6]], float)
    model = R.fit(d, g)
    assert model["rank"] == 1
    result = R.predict(model, [1, 7])
    np.testing.assert_allclose(result["predicted"], [2])
    np.testing.assert_allclose(result["unsupported"], [0, 7], atol=1e-12)


def test_source_equivalent_operators_cold_difference():
    d = np.array([[1, 2], [0, 0]], float)
    g = np.array([[2, 4]], float)
    model = R.fit(d, g)
    base, alternative = np.array([[2, 0.0]]), np.array([[2, 5.0]])
    np.testing.assert_array_equal(base @ d, alternative @ d)
    assert R.predict(model, [0, 1])["predicted"][0] == 0
    assert (alternative @ [0, 1])[0] == 5


def test_zero_support_keeps_source_failure():
    model = R.fit(np.zeros((3, 2)), np.ones((2, 2)))
    assert model["rank"] == 0
    assert model["source_interpolation_error"] == 2
    np.testing.assert_array_equal(R.predict(model, [1, 2, 3])["predicted"], [0, 0])


def test_non_linear_counterexample_not_rescued():
    d = np.array([[1.0, -1]])
    g = d**2
    model = R.fit(d, g)
    assert model["source_interpolation_error"] > 1
    assert abs(R.predict(model, [2])["predicted"][0] - 4) > 3


def test_paired_gradient_and_interactions():
    cube = np.zeros((2, 2, 2, 2, 2))
    for h in range(2):
        for b in range(2):
            for p in range(2):
                for q in range(2):
                    cube[h, b, p, q] = (-1) ** p * np.array(
                        [2, (-1) ** h + 2 * (-1) ** b * (-1) ** q]
                    )
    common, fixed = R.paired_gradient(cube)
    np.testing.assert_array_equal(common, [2, 0])
    assert np.linalg.norm(fixed - common) > 0


def test_error_decomposition_signed_cancellation():
    values = R.error_parts([3, 4], [1, 2], [2, 5], [0.2, -0.1], 0.09)
    assert values["identity_error"] < 1e-15
    assert values["operator_error"] > 0
    assert values["interaction_error"] > 0
    assert values["finite_error"] < 0


def test_orthogonal_frame_transport():
    rng = np.random.default_rng(340)
    d, g, x = rng.normal(size=(8, 5)), rng.normal(size=(6, 5)), rng.normal(size=8)
    frame, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    output_frame, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    original = R.predict(R.fit(d, g), x)["predicted"]
    transformed = R.predict(R.fit(frame @ d, output_frame @ g), frame @ x)["predicted"]
    np.testing.assert_allclose(transformed, output_frame @ original, atol=1e-11)


def records():
    """Create eight source-bound records, not a source label on a mixed pool."""
    return [
        {
            "role": "source",
            "pair": p,
            "wording": w,
            "query": "q",
            "d": np.array([p, w]),
            "g": np.array([p + w]),
        }
        for p in (0, 1)
        for w in range(4)
    ]


def test_source_isolation():
    data = records()
    d, g = R.source_columns(data)
    assert d.shape == (2, 8) and g.shape == (1, 8)
    data[-1]["role"] = "application"
    with pytest.raises(ValueError, match="source-only"):
        R.source_columns(data)


def test_source_missing_or_duplicate():
    data = records()
    data[-1]["pair"] = 0
    with pytest.raises(ValueError, match="crossing"):
        R.source_columns(data)


def test_source_query_spaces_separate():
    data = records()
    data[-1]["query"] = "other"
    with pytest.raises(ValueError, match="concatenated"):
        R.source_columns(data)


def test_token_binding():
    left = {
        "pair": 0,
        "wording": 0,
        "query": "q",
        "role": "source",
        "early_query_side": 0,
        "early_query_token_ids": [4, 7, 4],
    }
    right = {**left, "early_query_side": 1}
    assert R.paired_tokens(left, right) == (4, 7, 4)
    right["early_query_token_ids"] = [4, 4, 7]
    with pytest.raises(ValueError, match="ordered"):
        R.paired_tokens(left, right)


@pytest.mark.parametrize("value", [[], [1, 2], [[np.nan]], [[np.inf]]])
def test_bad_columns(value):
    with pytest.raises(ValueError):
        R.columns(value)


def test_bad_input_dimensions():
    model = R.fit(np.eye(2), np.eye(2))
    with pytest.raises(ValueError):
        R.predict(model, [1])


def test_zero_statistic():
    result = R.statistics([0, 0], [1, -1])
    assert result["relative_rmse"] is None
    assert result["sign_agreement"] == 0
    assert result["rmse"] == 1
