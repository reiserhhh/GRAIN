#!/usr/bin/env python3
"""Finite observation contracts must preserve target evidence and signed effects."""

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.construction import math_absolute_context_observation as M
from grain_research.vector_mechanisms import direct_traits as D


@pytest.mark.parametrize("task", tuple(D.BIG5))
def test_original_target_tail(task):
    texts = [f"addressed text {j}" for j in range(12)]
    original, _ = D.render(texts, task)
    assert M.render(texts, task) == original
    for background in (texts, [f"other text {j}" for j in range(12)]):
        rendered = M.render(texts, task, background)
        assert rendered.endswith(original)
        assert rendered.count("Question: " + D.question(task)) == 1


def test_complete_histories_required():
    with pytest.raises(ValueError):
        M.render(["a"], "agreeableness")
    with pytest.raises(ValueError):
        M.render(["a"] * 12, "agreeableness", [""] * 12)


def test_registered_partner_no_self_no_shrink():
    p = M.partners()
    np.testing.assert_array_equal(p, np.roll(np.arange(24), -1))
    assert np.all(p != np.arange(24))
    with pytest.raises(ValueError):
        M.partners(23)


def test_readout_retains_noncompliance_and_is_gauge_invariant():
    z = np.arange(13.0)
    r = M.readouts(
        z[:10][None], np.array([logsumexp(z)]), np.array([12]), np.arange(10)
    )
    s = M.readouts(
        (z[:10] + 100)[None],
        np.array([logsumexp(z) + 100]),
        np.array([12]),
        np.arange(10),
    )
    for k in ("probabilities", "log_odds", "score", "candidate_mass"):
        np.testing.assert_allclose(r[k], s[k], atol=2e-14)
    assert not r["greedy_compliant"][0]
    assert 0 < r["candidate_mass"][0] < 1


def test_uniform_law_not_missing():
    r = M.readouts(np.zeros((1, 10)), np.log([11]), np.array([10]), np.arange(10))
    np.testing.assert_allclose(r["score"], 0.5)
    np.testing.assert_allclose(r["log_odds"], 0)
    np.testing.assert_allclose(r["candidate_mass"], 10 / 11)


def test_bad_readout_rejected():
    with pytest.raises(ValueError):
        M.readouts(np.zeros((1, 9)), np.zeros(1), np.zeros(1), np.arange(10))
    with pytest.raises(ValueError):
        M.readouts(np.zeros((1, 10)), np.zeros(1), np.zeros(1), np.arange(10))


def test_full_signed_cross_account():
    x = np.random.default_rng(230).normal(size=(24, 3, 5))
    a = M.accounts(x)
    np.testing.assert_allclose(a["pieces"].sum(1), x[:, 2], atol=1e-15)
    np.testing.assert_allclose(
        a["cross_covariance"].sum((0, 1)), a["other_covariance"], atol=1e-15
    )
    assert np.any(a["cross_covariance"] < 0)


def test_zero_background_effect():
    v = np.random.default_rng(1).normal(size=(24, 5))
    a = M.accounts(np.repeat(v[:, None], 3, axis=1))
    np.testing.assert_array_equal(a["pieces"][:, 1:], 0)


def test_geometry_missing_constant_not_zero_contact():
    a = M.geometry(np.zeros((24, 5)))
    assert np.isnan(a["cosine"]).all()
    assert a["effective_rank"] is None


def test_geometry_offset_invariance_and_negative_contact():
    x = np.random.default_rng(1).normal(size=(24, 5))
    x[:, 1] = -x[:, 0]
    a, b = M.geometry(x), M.geometry(x + np.arange(5) * 10)
    np.testing.assert_allclose(a["cosine"], b["cosine"], atol=1e-14)
    assert a["cosine"][0, 1] == pytest.approx(-1)


def test_bootstrap_keeps_paired_conditions():
    v = np.random.default_rng(1).normal(size=(24, 5))
    x = np.repeat(v[:, None], 3, axis=1)
    a = M.bootstrap_en(x, 21, 230)
    np.testing.assert_array_equal(a, M.bootstrap_en(x, 21, 230))
    np.testing.assert_array_equal(a[:, 3:], 0)
