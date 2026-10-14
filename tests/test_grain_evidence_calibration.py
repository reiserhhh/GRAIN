#!/usr/bin/env python3
"""Checks on ownership, polarity and finite-pool weighting, not private labels."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_grain_evidence_calibration import (
    assignment_sensitivity,
    directed_code,
    precision_summary,
    stratified_rate,
)


def test_scope_and_reverse_keys_are_not_absence():
    f = pd.DataFrame(
        {
            "self_attribution": ["self"] * 4 + ["other"],
            "simpa_category": [1, 2, 3, 4, 1],
        }
    )
    assert directed_code(f, "general").tolist() == [0, 1, 2, 2, 2]
    assert directed_code(f, "episode_inclusive").tolist() == [0, 1, 0, 1, 2]


def test_stratified_pool_rate_not_naive_sample_mean():
    f = pd.DataFrame({"sampling_weight": [90, 10]})
    assert stratified_rate(f, [0, 1]) == pytest.approx(0.1)
    assert precision_summary(f, [1, 1], [0, 1])["weighted_precision"] == pytest.approx(
        0.1
    )
    assert precision_summary(f, [0, 0], [0, 1])["weighted_precision"] is None


def test_no_evidence_stays_missing_during_error_propagation():
    e = pd.DataFrame(
        columns=["author_index", "accepted", "comment_index", "trait", "key"]
    )
    e["accepted"] = e.accepted.astype(bool)
    b = [{"official_folds": {"big5": 1}}]
    labels = pd.DataFrame({"self_attribution": ["self"], "simpa_category": [1]})
    rows, info = assignment_sensitivity(e, b, labels, scope="general", draws=20)
    assert np.all(rows.score_available_draw_fraction == 0)
    assert rows.conditional_draw_median.isna().all()
    assert info["retained_sentence_count_draw_interval"] == [0, 0, 0]
