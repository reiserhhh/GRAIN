#!/usr/bin/env python3
"""Finite observation-law tests, not personality validation."""

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.construction import math_event_anchor_field as M


def fixture():
    return {
        "source_comments": np.array(
            [[[-1.0, 0.0], [1.0, 0.0]] * 2, [[-2.0, 0.0], [2.0, 0.0]] * 2]
        ),
        "codes": np.array([31, 0]),
        "anchors": np.array([[0.8] * 5, [0.2] * 5]),
        "variance": 0.1,
        "directions": np.array([[2.0], [0.0]]),
    }


def test_same_mean_is_not_same_event_distribution():
    b = fixture()
    result = M.score_events(b, [[1.0, 0.0]])
    assert np.allclose(result["E5/window_mean"]["probability"], 0.5)
    assert np.all(result["E5/comment_mixture"]["probability"] > 0.99)


def test_gaussian_mixture_direct_arithmetic():
    b = fixture()
    x = np.array([[0.4, 0.2]])
    got = M.author_log_law(x, b["source_comments"], 0.1)
    expected = np.array(
        [
            np.log(np.exp(-np.sum((x[0] - row) ** 2, 1) / 0.2).mean())
            for row in b["source_comments"]
        ]
    )
    assert np.allclose(got[0], expected)


def test_nonorthogonal_projection_has_covariant_metric():
    d = np.array([[1.0, 0.4], [0.5, 1.0], [0.0, 0.3]])
    q = M.projection(d)
    assert np.allclose(q.T @ q, np.eye(2))
    x = np.array([0.2, -0.3, 0.4])
    assert np.isclose(np.sum((x @ q) ** 2), (x @ d) @ np.linalg.solve(d.T @ d, x @ d))


def test_full_observation_reparameterization_invariance():
    b = fixture()
    x = np.array([[0.2, 0.8]])
    d = np.array([[2.0, 0.5], [0.3, 1.0]])
    q = M.projection(d)
    assert np.allclose(
        M.author_log_law(x @ q, b["source_comments"] @ q, 0.1),
        M.author_log_law(x, b["source_comments"], 0.1),
    )


def test_prior_free_log_bayes_factor():
    c = np.array([31, 31, 0])
    a = np.ones((3, 5)) * 0.5
    out = M.readings(np.zeros((1, 3)), c, a)
    assert np.allclose(out["probability"], 2 / 3)
    assert np.allclose(out["log_Bayes_factor"], 0)
    assert np.allclose(out["profile_mass"].sum(1), 1)


def test_codes_exchange_changes_meaning_not_author_likelihood():
    b = fixture()
    ell = M.author_log_law([[1.0, 0.0]], b["source_comments"], 0.1)
    a, c = (
        M.readings(ell, b["codes"], b["anchors"]),
        M.readings(ell, b["codes"][::-1], b["anchors"][::-1]),
    )
    assert np.allclose(a["probability"] + c["probability"], 1)
    assert np.allclose(a["log_Bayes_factor"], -c["log_Bayes_factor"])


def test_shared_author_history_not_product_of_marginal_probabilities():
    b = fixture()
    r = {
        "frame_key": "new",
        "events": [
            {"event_id": "a", "vector": [1.0, 0.0]},
            {"event_id": "b", "vector": [-1.0, 0.0]},
        ],
    }
    out = M.cold_record(b, r)["readings"]["E5/comment_mixture"]
    ell = M.author_log_law([[1.0, 0.0], [-1.0, 0.0]], b["source_comments"], 0.1)
    expected = M.readings(ell.sum(0, keepdims=True), b["codes"], b["anchors"])
    assert np.allclose(out["probability"], expected["probability"][0])


@pytest.mark.parametrize("bad", ["label", "answer", "questionnaire"])
def test_cold_rejects_outcome_inputs(bad):
    r = {
        "frame_key": "new",
        "events": [{"event_id": "a", "vector": [1.0, 0.0]}],
        bad: 1,
    }
    with pytest.raises(ValueError):
        M.cold_record(fixture(), r)


def test_duplicate_event_rejected():
    with pytest.raises(ValueError):
        M.cold_record(
            fixture(),
            {
                "frame_key": "new",
                "events": [{"event_id": "a", "vector": [1.0, 0.0]}] * 2,
            },
        )


def test_missing_event_rejected():
    with pytest.raises(ValueError):
        M.cold_record(fixture(), {"frame_key": "new", "events": []})


def test_no_target_side_is_not_imputed():
    b = fixture()
    with pytest.raises(ValueError):
        M.validate(b["source_comments"], [31, 31], b["anchors"], 0.1)


def test_constant_anchor_remains_constant():
    b = fixture()
    ell = M.author_log_law([[1.0, 0.0], [3.0, 0.0]], b["source_comments"], 0.1)
    out = M.readings(ell, b["codes"], np.full((2, 5), 0.4))
    assert np.allclose(out["anchor_mean"], 0.4)


def test_source_author_weights_not_prototype_count_priors():
    b = fixture()
    z = np.repeat(b["source_comments"], 3, axis=1)
    assert np.allclose(
        M.author_log_law([[1.0, 0.0]], z, 0.1),
        M.author_log_law([[1.0, 0.0]], b["source_comments"], 0.1),
    )


def test_common_log_offset_cancels():
    ell = np.array([[0.1, 0.2]])
    b = fixture()
    a = M.readings(ell, b["codes"], b["anchors"])
    c = M.readings(ell - logsumexp(ell), b["codes"], b["anchors"])
    for name in a:
        assert np.allclose(a[name], c[name])
