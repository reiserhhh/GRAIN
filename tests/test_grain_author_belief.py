#!/usr/bin/env python3
"""Finite Bayesian observation laws and leakage-free cold evidence contracts."""

import copy

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.construction import math_author_belief as M


def bundle():
    """Positive three-state fixture, not a simulated psychological discovery."""
    c = np.array([[[4, 1, 2], [1, 2, 1], [3, 1, 2]], [[1, 5, 1], [3, 1, 2], [1, 4, 1]]])
    c0 = np.array([[2, 1, 1], [1, 2, 1]])
    base = np.full((3, 3), 1 / 3)
    base0 = np.full(3, 1 / 3)
    return {
        "alpha": c + base,
        "alpha0": c0 + base0,
        "count": c,
        "count0": c0,
        "base": base,
        "base0": base0,
        "codes": np.array([[0, 31], [31, 0]]),
    }


@pytest.mark.parametrize("kernel", [False, True])
def test_closed_form_and_two_episode_carry(kernel):
    b = bundle()
    episodes = [np.array([0, 1, 0, 2, 0, 1]), np.array([2, 1, 2, 0])]
    first = M.observe(b, episodes[0], True, kernel)
    second = M.observe(b, episodes[1], True, kernel, first["carry"])
    ell = M.component_log_likelihood(b, episodes, kernel)
    total = logsumexp(ell) - np.log(2)
    assert abs(first["logp"].sum() + second["logp"].sum() - total) < 1e-12
    np.testing.assert_allclose(
        second["carry"]["logw"], ell - logsumexp(ell), atol=1e-12
    )
    np.testing.assert_allclose(second["predictions"].sum(1), 1)
    np.testing.assert_allclose(
        second["probability_shares_source_prior_application"].sum(1), 1
    )


def test_programs_same_first_step_not_same_history():
    b, x = bundle(), np.array([0, 1, 0, 1, 0, 1, 2])
    rows = [M.observe(b, x, *program) for program in M.PROGRAMS]
    for r in rows[1:]:
        np.testing.assert_array_equal(r["predictions"][:2], rows[0]["predictions"][:2])
    assert (
        np.linalg.norm(rows[0]["predictions"][2:] - rows[1]["predictions"][2:]) > 0.01
    )
    assert (
        np.linalg.norm(rows[1]["predictions"][2:] - rows[3]["predictions"][2:]) > 0.01
    )
    assert not rows[0]["profile_is_author_posterior"]


@pytest.mark.parametrize("kernel", [False, True])
def test_same_count_order_equivalence(kernel):
    b = bundle()
    x, y = np.array([0, 1, 0, 2, 0, 1]), np.array([0, 2, 0, 1, 0, 1])
    np.testing.assert_allclose(
        M.component_log_likelihood(b, [x], kernel),
        M.component_log_likelihood(b, [y], kernel),
    )
    np.testing.assert_allclose(
        M.observe(b, x, True, kernel)["profiles"][:, -1],
        M.observe(b, y, True, kernel)["profiles"][:, -1],
    )


def test_source_support_not_prior_fabrication():
    x = np.array(
        [
            [[0, 0, 1, 1, 2, 2, 0, 2, 1, 0]],
            [[1, 1, 0, 0, 2, 2, 1, 2, 0, 1]],
            [[2, 2, 0, 0, 1, 1, 2, 1, 0, 2]],
        ]
    )
    y = np.zeros((2, 3, 5), int)
    b = M.compile_source(x, y, np.arange(3), states=3)
    np.testing.assert_allclose(b["alpha"].sum(2) - b["count"].sum(2), 1)
    with pytest.raises(ValueError):
        M.compile_source(x[:, :, :2], y, np.arange(3), states=3)


def test_labels_only_change_reading_not_path_law():
    b = bundle()
    other = {**b, "codes": b["codes"][:, ::-1]}
    x = np.array([0, 1, 0, 2])
    a, c = M.observe(b, x), M.observe(other, x)
    np.testing.assert_array_equal(a["predictions"], c["predictions"])
    assert np.linalg.norm(a["marginals"] - c["marginals"]) > 0.01


def test_cold_rejects_duplicates_labels_and_boundaries():
    r = {
        "author_key": "new",
        "episodes": [
            {
                "events": [
                    {"event_id": "a", "timestamp": 1.0, "cell": 0},
                    {"event_id": "b", "timestamp": 2.0, "cell": 1},
                ]
            },
            {"events": [{"event_id": "c", "timestamp": 3.0, "cell": 2}]},
        ],
    }
    assert len(M.validate_record(r, 3)) == 2
    assert len(M.cold(bundle(), r)["episodes"]) == 2
    for kind in ("duplicate", "label", "time"):
        bad = copy.deepcopy(r)
        if kind == "duplicate":
            bad["episodes"][1]["events"][0]["event_id"] = "a"
        elif kind == "label":
            bad["big5"] = [1] * 5
        else:
            bad["episodes"][1]["events"][0]["timestamp"] = 1.0
        with pytest.raises(ValueError):
            M.validate_record(bad, 3)


def test_stranger_donor_has_same_start_never_self_or_imputed_singleton():
    paths = np.array([[0, 1], [0, 2], [1, 0], [2, 0]])
    np.testing.assert_array_equal(M.start_matched_donors(paths), [1, 0, -1, -1])
