#!/usr/bin/env python3
"""Synthetic mechanism controls before any HM139 application read."""

import numpy as np
import pytest

from grain_research.construction import math_context_action_transport as M


def test_native_actor_and_fact_binding():
    y = np.arange(14 * 4).reshape(14, 4)
    z = M.person_channels(y)
    np.testing.assert_array_equal(z[0, 6], y[12])
    np.testing.assert_array_equal(z[1, 0], y[6])
    np.testing.assert_array_equal(z[1, 6], y[13])


def test_positive_author_laws():
    plus, minus = M.positive_laws()
    assert plus.min() > 0
    assert plus.sum() == minus.sum() == 1
    np.testing.assert_allclose(plus.sum(0), minus.sum(0))
    np.testing.assert_allclose(plus.sum(1), minus.sum(1))
    x = np.arange(12).reshape(2, 2, 3) ** 2
    np.testing.assert_allclose(
        np.einsum("bc,bck->k", plus - minus, x), M.interaction(x)
    )


def test_condition_action_not_global():
    x = np.array([[1.0], [1.0], [2.0], [2.0]])
    c = np.array([0, 1, 0, 1])
    y = (x[:, 0] * np.array([1, -1, 1, -1]))[:, None]
    global_model = M.fit_actions(x, y)
    conditional = M.fit_actions(M.features(x, c), y)
    assert global_model["source_error_rms"] > 1
    assert conditional["source_error_rms"] < 1e-12


def test_source_units_and_zero_features():
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(size=(18, 6)), np.zeros(18)])
    y = rng.normal(size=(18, 4))
    a, b = M.fit_actions(x, y), M.fit_actions(x * [1, 2, 3, 4, 5, 6, 1], y)
    np.testing.assert_allclose(
        x @ a["beta"], (x * [1, 2, 3, 4, 5, 6, 1]) @ b["beta"], atol=1e-12
    )
    assert a["alternate_prediction_error"] < 1e-12


def test_no_target_fields_or_unobserved_context():
    models = {
        "global": {"beta": np.zeros((15, 2)).tolist()},
        "real": {"beta": np.zeros((30, 2)).tolist()},
        "sham": {"beta": np.zeros((30, 2)).tolist()},
    }
    r = {"g": np.ones(15), "real_context": 0, "sham_context": 1}
    assert set(M.cold_score({"models": models}, [r])) == {
        "global",
        "real",
        "sham",
        "wrong_real",
    }
    with pytest.raises(ValueError):
        M.cold_score({"models": models}, [{**r, "family": 2}])
    with pytest.raises(ValueError):
        M.cold_score({"models": models}, [{**r, "real_context": None}])


def test_sham_has_no_outcome_dependency():
    a = M.balanced_labels(24, 139)
    assert np.sum(a) == 12
    np.testing.assert_array_equal(a, M.balanced_labels(24, 139))
    assert not np.array_equal(a, M.balanced_labels(24, 140))


def test_zero_and_signed_metrics():
    assert M.compare(np.zeros(4), np.zeros(4))["cosine"] is None
    assert M.compare(-np.ones(4), np.ones(4))["signed_projection"] == -1
    with pytest.raises(ValueError):
        M.interaction(np.ones((2, 1)))
