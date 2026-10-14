#!/usr/bin/env python3
"""Fast semantic-contract checks; no model download or private author text."""

import numpy as np

from scripts.run_grain_paraphrase_reference import contrasts, detection_geometry


def test_semantic_fixture_labels_are_not_inherited_from_word_overlap():
    rows = contrasts()
    assert len(rows) == 40
    for category in ("equivalent", "opposite", "other_person", "single_episode"):
        subset = [r for r in rows if r["category"] == category]
        assert len(subset) == 10
        assert all(
            r["exact_habitual_self_match_expected"] == (category == "equivalent")
            for r in subset
        )
    # This common prefilter deliberately fails to establish psychological ownership.
    assert all("I " in r["sentence"] for r in rows if r["category"] == "other_person")


def test_geometry_measures_acceptance_not_annotation_accuracy():
    q = np.eye(3)
    h = np.array([[1.0, 0, 0], [0, 1, 0], [-1.0, 0, 0]])
    result = detection_geometry(np.vstack([q, h]), 3)
    assert result["top1_acceptance_at_0_6"] == 2 / 3
    assert 0 <= result["random_pair_acceptance_at_0_6"] <= 1
    assert "accuracy" not in result and "false_positive_rate" not in result


def test_sparse_evidence_does_not_get_a_zero_width_confidence_certificate():
    from scripts.audit_grain_evidence_sampling import sampling_interval

    counts = np.zeros((12, 2, 2))
    counts[0, 0, 0] = 5
    rows = sampling_interval(counts)
    assert rows[0]["point"] == 1
    assert rows[0]["resampling_lower"] is None and rows[0]["resampling_upper"] is None
    assert rows[1]["point"] is None
    counts[:6, 1, 0] = 1
    rows = sampling_interval(counts)
    assert rows[1]["status"] == "degenerate_empirical_interval"
    counts[6:, 1, 1] = 1
    rows = sampling_interval(counts)
    assert rows[1]["resampling_lower"] < rows[1]["point"] < rows[1]["resampling_upper"]
