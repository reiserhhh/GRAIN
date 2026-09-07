#!/usr/bin/env python3
"""Check seminar counterexamples, not real-world psychological hypotheses."""

from itertools import pairwise

import pytest

from scripts.check_grain_seminar_witnesses import evidence_scores, witnesses


def test_residual_correlation_can_increase_while_contribution_decreases():
    c = witnesses()["residual_correlation"]
    assert c["before"] == pytest.approx(4 / 20**0.5)
    assert c["after_removing_x"] == pytest.approx(1)
    assert c["fixed_denominator_remaining"] == pytest.approx(1 / 20**0.5)
    assert c["closure_error"] < 1e-12


def test_current_score_compression_does_not_preserve_future_observability():
    c = witnesses()["observability"]
    assert c["current_outputs"] == [1, 1]
    assert c["next_outputs"] == [1, -1]
    assert c["rank_one_observation"] == 1
    assert c["rank_two_observations"] == 2
    assert c["minimum_next_mse_from_current_scalar"] == pytest.approx(1)


def test_duplication_does_not_identify_averaging_versus_dependent_bayes():
    w = witnesses()
    c = w["evidence_arithmetic"]
    for key in ("content_mean", "source_aware_lr_sum"):
        assert c["base"][key] == pytest.approx(c["all_copied_same_source"][key])
    assert c["same_values_independent_sources"]["source_aware_lr_sum"] == pytest.approx(
        4.8
    )
    assert c["all_copied_same_source"]["source_aware_lr_sum"] == pytest.approx(1.2)
    overlap = w["margin_interactions"]["same_source_overlapping_halves"]
    assert overlap["content_mean"] == pytest.approx(-0.8)
    assert overlap["source_aware_lr_sum"] == pytest.approx(-0.8)
    assert overlap["naive_lr_sum"] == pytest.approx(0)


def test_finite_world_rejects_undefined_provenance_cases():
    with pytest.raises(ValueError, match="agree"):
        evidence_scores([("same", 1.0), ("same", -1.0)])
    with pytest.raises(ValueError, match="finite"):
        evidence_scores([("a", float("nan"))])
    assert all(v == 0 for v in evidence_scores([]).values())


def test_two_distinct_normalization_rules_both_saturate():
    rows = witnesses()["normalization_competition"]
    assert [r["mean_with_fixed_null_slot"] for r in rows] == pytest.approx(
        [0.5, 2 / 3, 0.8, 8 / 9]
    )
    for key in ("mean_with_fixed_null_slot", "additive_state_l2_normalized_readout"):
        values = [r[key] for r in rows]
        assert all(0 < a < b < 1 for a, b in pairwise(values))
    assert rows[0]["mean_with_fixed_null_slot"] != pytest.approx(
        rows[0]["additive_state_l2_normalized_readout"]
    )
