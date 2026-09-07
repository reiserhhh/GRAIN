#!/usr/bin/env python3
"""Closed-form finite checks for comment-mass and conditional-moment readouts."""

import numpy as np

from grain_research.vector_mechanisms.conditional_moments import (
    conditional_moment_readouts,
)


def test_conditional_moment_arms_match_two_symmetric_groups():
    k = np.array([-1.0, 1.0, -2.0, 2.0, 0.0])[:, None]
    v = np.array([-1.0, 1.0, -3.0, 3.0, 100.0])[:, None]
    q = 0.7
    arms = conditional_moment_readouts(k, v, np.array([q]), [[0, 1], [2, 3]])
    z = 2 * np.cosh(q) + 2 * np.cosh(2 * q) + 1
    expected = {
        "exact": (2 * np.sinh(q) + 6 * np.sinh(2 * q)) / z,
        "uniform_moment": 2.8 * q,
        "exact_mass_moment": (2 * np.cosh(q) * q + 12 * np.cosh(2 * q) * q) / z,
        "gaussian_mass_moment": (2 * np.exp(q**2 / 2) * q + 12 * np.exp(2 * q**2) * q)
        / (2 * np.exp(q**2 / 2) + 2 * np.exp(2 * q**2) + 1),
    }
    for key, result in arms.items():
        np.testing.assert_allclose(result[0], expected[key], atol=1e-14)


def test_moment_arms_have_the_same_zero_query_derivative():
    rng = np.random.default_rng(938)
    k, v = rng.normal(size=(25, 5)), rng.normal(size=(25, 3))
    groups = [list(range(1, 12)), list(range(13, 20))]
    zero = conditional_moment_readouts(k, v, np.zeros(5), groups)
    for result in zero.values():
        np.testing.assert_allclose(result, 0, atol=1e-14)
    direction, eps = rng.normal(size=5), 1e-5
    left = conditional_moment_readouts(k, v, -eps * direction, groups)
    right = conditional_moment_readouts(k, v, eps * direction, groups)
    slope = (right["exact"] - left["exact"]) / (2 * eps)
    for name in right:
        np.testing.assert_allclose(
            (right[name] - left[name]) / (2 * eps), slope, atol=1e-8
        )


def test_multihead_moment_exact_arm_replays_prefix_pairing():
    from grain_research.vector_mechanisms.prefix import prefix_pairing
    from scripts.run_grain_conditional_moments import ARMS, head_readouts

    rng = np.random.default_rng(939)
    cache = {
        "q": rng.normal(size=(4, 6)),
        "k": rng.normal(size=(20, 2, 6)),
        "v": rng.normal(size=(20, 2, 6)),
    }
    groups = [[1, 2, 3, 4], [7, 8, 9, 10, 11]]
    arms = head_readouts(cache, groups, cache["q"])
    assert tuple(arms) == ARMS
    assert all(vector.shape == (24,) for vector in arms.values())
    np.testing.assert_allclose(arms["exact"], prefix_pairing(cache, groups), atol=1e-14)
