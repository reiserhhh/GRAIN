#!/usr/bin/env python3
"""Mechanism tests for joint author accounts, not prediction accuracy."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_author_observation_accounts as M


def laws():
    """Independent synthetic laws with six vector outputs."""
    return np.random.default_rng(69).normal(size=(2, 8, 6))


def test_disintegration_preserves_joint_distribution():
    p = np.arange(1.0, 9.0) / 36
    assert np.allclose(M.profile(*M.disintegrate(p)), p)


@pytest.mark.parametrize("case", M.CASES)
def test_all_declared_cases_have_complete_probability_laws_and_exact_accounts(case):
    f = laws()
    p0, p1, a, b, _ = M.case_profiles(case, *f)
    for p in (p0, p1):
        assert p.min() > 0
        assert p.sum() == pytest.approx(1)
    result = M.accounts(p0, p1, f[a], f[b])
    assert np.allclose(result["shapley"].sum(0), result["delta"])
    assert np.allclose(result["sequences"].sum(1), result["delta"])
    assert np.allclose(result["sequences"].mean(0), result["shapley"])


@pytest.mark.parametrize(
    "case,index", [("choice_f0", 0), ("expression_f0", 1), ("law_only", 2)]
)
def test_pure_cases_assign_zero_to_unchanged_factors(case, index):
    f = laws()
    p0, p1, a, b, _ = M.case_profiles(case, *f)
    r = M.accounts(p0, p1, f[a], f[b])
    assert np.allclose(r["shapley"][index], r["delta"])
    assert np.allclose(np.delete(r["shapley"], index, axis=0), 0)


def test_null_profiles_keep_all_six_scores_but_change_conditional_profiles():
    p0, p1, rank = M.equal_score_profiles(laws()[0])
    assert rank == 7
    assert np.allclose(p0 @ laws()[0], p1 @ laws()[0])
    pi0, mu0 = M.disintegrate(p0)
    pi1, mu1 = M.disintegrate(p1)
    assert not np.allclose(pi0, pi1)
    assert not np.allclose(mu0, mu1)


def test_sequence_allocation_can_differ_while_symmetric_account_is_fixed():
    f = laws()
    p0, p1, a, b, _ = M.case_profiles("mixed", *f)
    r = M.accounts(p0, p1, f[a], f[b])
    assert np.ptp(r["sequences"], axis=0).max() > 0.01
    assert np.allclose(r["sequences"].mean(0), r["shapley"])


def test_parity_expression_profiles_keep_all_pairwise_marginals():
    p0, p1, *_ = M.case_profiles("expression_f0", *laws())
    xyz = np.array(list(itertools.product((-1, 1), repeat=3)))
    for d in (1, 2):
        for columns in itertools.combinations(range(3), d):
            character = xyz[:, columns].prod(1)
            assert (p1 - p0) @ character == pytest.approx(0)
    assert (p1 - p0) @ xyz.prod(1) == pytest.approx(1)


@pytest.mark.parametrize("p", [np.zeros(8), np.full(8, np.nan), np.full(7, 1 / 7)])
def test_missing_nonfinite_invalid_profile_fails(p):
    with pytest.raises(ValueError):
        M.disintegrate(p)
