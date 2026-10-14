#!/usr/bin/env python3
"""Exact conditional likelihood and cold-input boundaries for HM140."""

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import binom

from grain_research.construction import math_relative_choice as M


@pytest.mark.parametrize("bias", [-4.0, -0.5, 2.5])
@pytest.mark.parametrize("baseline", [-1.7, 0.0, 1.2])
def test_direct_joint_matches_conditional_law(bias, baseline):
    """Shared bias and nonzero reference level cancel by direct enumeration."""
    support, log_mass, _, _ = M.conditional_law(7, 9, 8, 1.1)
    joint = binom.pmf(support, 7, expit(bias + baseline + 1.1))
    joint *= binom.pmf(8 - support, 9, expit(bias + baseline))
    assert np.allclose(joint / joint.sum(), np.exp(log_mass), atol=1e-14)


def test_variance_is_negative_loglikelihood_curvature():
    """An independently differenced partition function recovers information."""
    a = 0.7
    _, log_mass, _, var = M.conditional_law(8, 9, 7, a)
    eps = 2e-4
    minus = M.conditional_law(8, 9, 7, a - eps)[1]
    plus = M.conditional_law(8, 9, 7, a + eps)[1]
    assert np.max(abs(-(plus - 2 * log_mass + minus) / eps**2 - var)) < 1e-6


def record(**changes):
    """Public count-only fixture with explicit event qualification."""
    return {
        "menu": 0,
        "n": 8,
        "success": 6,
        "ref_n": 8,
        "ref_success": 2,
        "qualification": "free",
        **changes,
    }


def test_forced_and_missing_support_do_not_fabricate_measurements():
    d, grid = [[1.0]], [[-1.0], [0.0], [1.0]]
    forced = M.cold_score([record(qualification="forced")], d, grid)
    missing = M.cold_score([record(ref_n=0, ref_success=0)], d, grid)
    for score in (forced, missing):
        assert score["status"] == "NO_IDENTIFYING_CHOICE"
        assert np.allclose(score["posterior"], 1 / 3)
        assert np.max(abs(score["information"])) == 0


@pytest.mark.parametrize(
    "change",
    [
        {"truth": 0.2},
        {"author_id": "x"},
        {"qualification": "unknown"},
        {"success": 8.1},
        {"n": -1},
        {"menu": 1},
        {"ref_success": 9},
    ],
)
def test_unqualified_or_leaking_input_rejected(change):
    with pytest.raises(ValueError):
        M.cold_score([record(**change)], [[1.0]], [[-1.0], [1.0]])


def test_duplicate_menu_rejected():
    with pytest.raises(ValueError):
        M.cold_score([record(), record()], [[1.0]], [[-1.0], [1.0]])


def test_all_equal_margin_has_no_conditional_information():
    score = M.cold_score([record(success=0, ref_success=0)], [[1.0]], [[-1.0], [1.0]])
    assert score["status"] == "NO_IDENTIFYING_CHOICE"
    assert np.max(abs(score["information"])) == 0


def test_menu_null_and_common_gauge_are_observational_equivalences():
    d = np.array([[1.0, 2.0, 0.0], [2.0, -1.0, 0.0]])
    g = M.geometry(d)
    assert g["rank"] == 2 and g["null"].shape == (3, 1)
    theta, bias, shift = np.array([0.3, 0.7, -0.9]), np.array([1.1, -0.4]), np.ones(3)
    assert np.allclose(
        expit(bias + d @ theta), expit(bias - d @ shift + d @ (theta + shift))
    )
    assert np.max(abs(d @ g["null"])) < 1e-14


def test_personal_context_alias_cannot_be_removed_as_shared_bias():
    d = np.array([[1.0, 0.5], [-0.4, 1.0]])
    theta, common = np.array([0.7, -0.3]), np.array([0.5, 0.2])
    personality_world = expit(common + d @ theta)
    personal_context_world = expit(common + d @ np.zeros(2) + d @ theta)
    assert np.array_equal(personality_world, personal_context_world)


def test_utility_temperature_scale_is_not_identified_by_probability():
    utility = np.array([-0.3, 0.7, 1.2])
    assert np.array_equal(expit(1.7 * utility), expit(3.4 * utility / 2))
