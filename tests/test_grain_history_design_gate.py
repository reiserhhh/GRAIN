#!/usr/bin/env python3
"""Observation-law eligibility checks, not an accuracy search."""

from fractions import Fraction as F

from grain_research.construction import math_history_design_gate as G
from grain_research.construction import math_history_resolution as M


def test_doubling_requires_shared_independent_parameters():
    p, even, odd = M.matched_populations()
    for mode in ("shared_independent", "distinct_parameters", "copied_channel"):
        left, right = G.history_law(p, even, mode), G.history_law(p, odd, mode)
        assert sum(left) == sum(right) == 1
        assert (left == right) == (mode != "shared_independent")


def test_observed_parameter_gap_and_copy_excess():
    p, mass, _ = M.matched_populations()
    independent = G.source_observable_checks(
        G.history_law(p, mass, "shared_independent")
    )
    assert all(value == 0 for value in independent.values())
    distinct = G.source_observable_checks(G.history_law(p, mass, "distinct_parameters"))
    copied = G.source_observable_checks(G.history_law(p, mass, "copied_channel"))
    assert distinct["cross_event_parameter_gap"] == F(1, 20)
    assert copied["within_event_excess_covariance"] == F(1, 5)


def test_zero_pooled_covariance_does_not_prove_independence():
    p, mass = [F(1, 4), F(3, 4)], [F(1, 2), F(1, 2)]
    law = G.history_law(p, mass, "balanced_dependence")
    checks = G.source_observable_checks(law)
    assert checks["cross_event_parameter_gap"] == 0
    assert checks["within_event_excess_covariance"] == 0
    assert checks["choice_weighted_excess_covariance"] == F(1, 200)
    assert law != G.history_law(p, mass, "shared_independent")
