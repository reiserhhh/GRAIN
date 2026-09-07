#!/usr/bin/env python3
"""Finite gain/allocation identities and no-fit second-order checks."""

import numpy as np

from grain_research.vector_mechanisms.conditional_moments import (
    conditional_moment_readouts,
)
from grain_research.vector_mechanisms.prefix import prefix_pairing
from grain_research.vector_mechanisms.readout_gain import gain_readouts, multihead_gain


def test_gain_arms_replay_exact_and_two_prior_moment_arms():
    rng = np.random.default_rng(940)
    k, v, q = rng.normal(size=(21, 5)), rng.normal(size=(21, 3)), rng.normal(size=5)
    groups = [[1, 2, 3, 4], [8, 9, 10, 11, 12, 13]]
    old = conditional_moment_readouts(k, v, q, groups)
    arms, d = gain_readouts(k, v, q, groups)
    for new, previous in (
        ("exact", "exact"),
        ("exact_mass_cov", "exact_mass_moment"),
        ("uniform_cov", "uniform_moment"),
    ):
        np.testing.assert_allclose(arms[new], old[previous], atol=1e-14)
    np.testing.assert_allclose(sum(d["group_mass"]), d["body_mass"])
    np.testing.assert_allclose(d["body_mass"] + d["outside_mass"], 1)
    assert 0 <= d["normalized_comment_entropy"] <= 1


def test_single_group_removes_allocation_difference_and_second_order_is_local():
    rng = np.random.default_rng(941)
    k, v, q = rng.normal(size=(17, 4)), rng.normal(size=(17, 3)), rng.normal(size=4)
    groups = [list(range(2, 13))]
    arms, detail = gain_readouts(k, v, q * 1e-3, groups)
    np.testing.assert_allclose(arms["gain_only_exact"], arms["exact"], atol=1e-14)
    np.testing.assert_allclose(
        arms["gain_only_cov"], arms["exact_mass_cov"], atol=1e-14
    )
    assert detail["normalized_comment_entropy"] == 0
    assert np.linalg.norm(
        arms["exact_mass_second"] - arms["exact"]
    ) < 0.01 * np.linalg.norm(arms["exact_mass_cov"] - arms["exact"])


def test_head_contributions_reconstruct_score_and_constant_values_have_no_signal():
    rng = np.random.default_rng(942)
    cache = {
        "q": rng.normal(size=(4, 6)),
        "k": rng.normal(size=(20, 2, 6)),
        "v": rng.normal(size=(20, 2, 6)),
    }
    groups, beta = [[1, 2, 3, 4], [9, 10, 11, 12]], rng.normal(size=24)
    arms, detail = multihead_gain(cache, groups, cache["q"], beta)
    np.testing.assert_allclose(arms["exact"], prefix_pairing(cache, groups), atol=1e-14)
    for arm, vector in arms.items():
        np.testing.assert_allclose(
            vector @ beta, sum(d["contributions"][arm] for d in detail), atol=1e-14
        )
    constant, _ = gain_readouts(
        cache["k"][:, 0], np.ones((20, 6)), cache["q"][0], groups
    )
    for vector in constant.values():
        np.testing.assert_array_equal(vector, np.zeros(6))
