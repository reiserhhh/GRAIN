#!/usr/bin/env python3
"""Test finite algebra and its scope guards, not a learned personality model."""

import numpy as np
import pytest

from scripts.check_grain_evidence_state_theory import (
    anchored_mean,
    append_evidence,
    covariance_terms,
    cross_ratio,
    query_moments,
    witnesses,
)


def test_fixed_attention_and_homogeneous_coordinates_are_same_readout():
    w = witnesses()
    assert w["frozen_attention"]["maximum_identity_error"] < 1e-12
    assert w["anchored_curve"]["additive_moment_ratio_error"] < 1e-12
    assert w["anchored_curve"]["common_mass_scale_error"] < 1e-12


def test_relative_mass_and_projective_invariant_have_exact_recovery():
    c = witnesses()["anchored_curve"]
    assert c["recovered_ratios"] == pytest.approx([c["ratio"]] * 4)
    assert c["cross_ratio_output"] == pytest.approx(c["cross_ratio_input"])
    assert abs(c["l2_cross_ratio"] - c["cross_ratio_input"]) > 0.01
    reversed_curve = anchored_mean([1, 2, 4, 8], ratio=0.7, baseline=0.9, evidence=-0.2)
    assert cross_ratio(reversed_curve) == pytest.approx(c["cross_ratio_input"])


def test_saturation_and_degenerate_curves_are_not_false_identification():
    values = anchored_mean([0, 1, 10], ratio=1, baseline=0.5, evidence=0.5)
    assert values.tolist() == [0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match="distinct"):
        cross_ratio([0.5] * 4)
    with pytest.raises(ValueError, match="positive"):
        anchored_mean([1], ratio=0, baseline=0, evidence=1)
    with pytest.raises(ValueError, match="nonnegative"):
        anchored_mean([-1], ratio=1, baseline=0, evidence=1)
    with pytest.raises(ValueError, match="finite"):
        cross_ratio([1, 2, 3, float("nan")])


def test_equal_scores_need_mass_to_predict_the_next_update():
    c = witnesses()["update_state"]
    assert c["next_means"] == pytest.approx([0.75, 0.55])
    assert c["batch_stream_error"] < 1e-12
    z, t = append_evidence(2.0, [1.0], 0.0, [9.0])
    assert z == 2.0 and t[0] == 1.0
    with pytest.raises(ValueError, match="Positive"):
        append_evidence(0.0, [0.0], 1, [1.0])
    with pytest.raises(ValueError, match="shapes"):
        append_evidence(1.0, [0.0, 0.0], 1, [1.0])


def test_moments_alone_do_not_identify_duplicate_sources():
    a, b = witnesses()["provenance"]["after_observation_a"]
    assert a == {"mass": 2.0, "mean": 0.5}
    assert b["mass"] == 3.0 and b["mean"] == pytest.approx(2 / 3)


def test_positive_kernel_moments_reproduce_multiple_queries():
    c = witnesses()["query_moments"]
    assert np.allclose(c["direct"], c["compressed"], atol=1e-12)
    assert c["maximum_identity_error"] < 1e-12
    with pytest.raises(ValueError, match="Nonnegative kernel"):
        query_moments([[1.0]], [[-1.0]], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="Incompatible"):
        query_moments([[1.0, 1.0]], [[1.0]], [[1.0]], [1.0])


def test_uniform_positive_gate_cannot_improve_pearson_but_changes_absolute_error():
    c = witnesses()["external_association"]
    assert c["constant_gate_correlation_before"] == pytest.approx(2 / np.sqrt(5))
    assert c["constant_gate_correlation_after"] == pytest.approx(
        c["constant_gate_correlation_before"]
    )
    assert c["mae_before"] == pytest.approx(0.4)
    assert c["mae_after"] == pytest.approx(0.76)
    assert c["constant_content_coverage_only_correlation"] == pytest.approx(1)
    assert c["constant_content_variance"] == 0


def test_variable_gate_covariance_closes_including_interaction():
    c = witnesses()["external_association"]["varying_gate_covariance"]
    assert c["content"] == pytest.approx(0)
    assert c["coverage"] == pytest.approx(0.1)
    assert c["interaction"] == pytest.approx(0.1)
    assert c["total"] == pytest.approx(0.2)
    assert c["closure_error"] < 1e-12
    with pytest.raises(ValueError, match="gates"):
        covariance_terms([0, 1], [0, 1], [0, 1.1], 0)


def test_covariance_identity_on_seeded_arbitrary_vectors():
    rng = np.random.default_rng(731)
    for _ in range(10):
        y, b = rng.normal(size=(2, 31))
        c = covariance_terms(y, b, rng.uniform(size=31), float(rng.normal()))
        assert c["closure_error"] < 1e-12
