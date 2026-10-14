#!/usr/bin/env python3
"""Synthetic and analytic checks for the declared common-reference mechanism."""

import numpy as np
import pytest
from scipy.integrate import quad

from grain_research.construction import math_shared_reference as M


def test_binary_closed_form():
    p, q = np.array([0.7, 0.3]), np.array([0.4, 0.6])
    np.testing.assert_allclose(M.coupling(p, q), [[0.4, 0.3], [0, 0.3]])


def test_self_swap_and_source_permutation():
    p, q = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.15, 0.35, 0.1])
    j = M.coupling(p, q)
    np.testing.assert_allclose(j.sum(1), p, atol=1e-14)
    np.testing.assert_allclose(j.sum(0), q, atol=1e-14)
    np.testing.assert_allclose(M.coupling(p, p), np.diag(p), atol=1e-14)
    np.testing.assert_allclose(M.coupling(q, p), j.T, atol=1e-14)
    ix = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(M.coupling(p[ix], q[ix]), j[np.ix_(ix, ix)])


def test_off_diagonal_against_exponential_integral():
    p, q = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.15, 0.35, 0.1])
    r, j = p / q, M.coupling(p, q)
    for i in range(4):
        for k in range(4):
            if r[i] > r[k]:
                integral = quad(
                    lambda t: 1 / np.maximum(p, t * q).sum() ** 2,
                    r[k],
                    r[i],
                    points=r[(r > r[k]) & (r < r[i])],
                )[0]
                assert j[i, k] == pytest.approx(p[i] * q[k] * integral, abs=1e-13)


def test_joint_anchor_means_and_conditional_comparison():
    p, q = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.15, 0.35, 0.1])
    y = np.arange(20).reshape(4, 5) / 20
    a = M.accounts(p, q, y)
    assert a["mean_difference_error"] < 1e-14
    np.testing.assert_allclose(
        a["probability"],
        0.5 * a["same_source_probability"]
        + a["changed_source_probability"]
        * a["conditional_probability_given_changed_source"],
    )
    self_account = M.accounts(p, p, y)
    np.testing.assert_allclose(self_account["probability"], 0.5)
    assert np.isnan(self_account["conditional_probability_given_changed_source"]).all()


def test_shared_race_agrees_with_pair_law_and_has_a_triple_law():
    w = np.array([[0.49, 0.49, 0.02], [0.02, 0.49, 0.49], [0.49, 0.02, 0.49]])
    e = np.random.default_rng(231).exponential(size=(50000, 3))
    samples = M.sample_shared(w, e)
    assert samples.shape == (50000, 3)
    counts = np.bincount(samples[:, 0] * 3 + samples[:, 1], minlength=9)
    empirical = counts / 50000
    np.testing.assert_allclose(
        empirical.reshape(3, 3), M.coupling(w[0], w[1]), atol=0.006
    )
    triples = np.bincount(samples @ np.array([9, 3, 1]), minlength=27).reshape(3, 3, 3)
    np.testing.assert_array_equal(triples.sum(2), counts.reshape(3, 3))


@pytest.mark.parametrize("bad", [[0, 1], [-0.1, 1.1], [0.2, 0.2], [np.nan, 0.5]])
def test_invalid_mass_is_not_repaired(bad):
    with pytest.raises(ValueError):
        M.coupling(bad, [0.5, 0.5])
