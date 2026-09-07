#!/usr/bin/env python3
"""Algebra, not a psychometric validity or newly fitted personality test."""

import numpy as np
import pytest

from scripts.check_grain_cue_utilization_theory import finite_worlds, four_terms


def test_known_alignment_worlds():
    c = finite_worlds()
    for name, expected in (("aligned", 0.64), ("orthogonal", 0), ("reversed", -0.64)):
        assert c[name]["R_y"] == pytest.approx(0.8)
        assert c[name]["R_s"] == pytest.approx(0.8)
        assert c[name]["actual_correlation"] == pytest.approx(expected, abs=1e-12)
        assert c[name]["closure_absolute_error"] < 1e-12
    for key in ("R_y", "R_s", "G", "actual_correlation"):
        assert c["aligned"][key] == pytest.approx(
            c["invertible_cue_coordinates"][key], abs=1e-12
        )


def test_residual_assignment_depends_on_cue_basis_not_truth():
    c = finite_worlds()
    a, b = c["omitted_shared_cue"], c["shared_cue_included"]
    assert a["actual_correlation"] == pytest.approx(b["actual_correlation"])
    assert a["unmodeled_alignment"] == pytest.approx(0.96)
    assert b["unmodeled_alignment"] == pytest.approx(0, abs=1e-12)


def test_shrinkage_breaks_two_term_simplification_but_not_accounting():
    c = finite_worlds()["shrinkage_requires_cross_terms"]
    assert c["actual_correlation"] == pytest.approx(1)
    assert c["two_term_omission_error"] == pytest.approx(0.5)
    assert c["modeled_y_unmodeled_s"] == pytest.approx(0.25)
    assert c["unmodeled_y_modeled_s"] == pytest.approx(0.25)
    assert c["closure_absolute_error"] < 1e-12


def test_arbitrary_prediction_identity_and_input_validation():
    rng = np.random.default_rng(101)
    a = rng.normal(size=(4, 100))
    assert four_terms(*a)["closure_absolute_error"] < 1e-12
    with pytest.raises(ValueError, match="constant"):
        four_terms(np.ones(100), *a[1:])
    with pytest.raises(ValueError, match="same paired"):
        four_terms(a[0][:-1], *a[1:])


def test_probability_and_duplicate_worlds():
    c = finite_worlds()
    b = c["binary_resolution_without_threshold_crossing"]
    assert b["expected_brier"] == pytest.approx(0.245)
    assert b["brier_decomposition_error"] < 1e-12
    assert b["expected_brier"] < b["uncertainty"]
    assert b["population_auc_with_ties"] == pytest.approx(0.8)
    assert b["balanced_accuracy_at_half"] == pytest.approx(0.5)
    d = c["duplicate_evidence"]
    assert (
        d["posterior_after_one_observation"]
        == d["posterior_after_perfect_duplicate_correctly_conditioned"]
    )
    assert d["posterior_if_duplicate_is_wrongly_independent"] == pytest.approx(16 / 17)
