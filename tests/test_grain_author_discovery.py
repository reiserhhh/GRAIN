#!/usr/bin/env python3
"""Minimum correctness for discovery-stage author and fixed-score comparisons."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts.analyze_grain_author_discovery import direct_prediction, scalar_metrics


def test_continuous_metrics_and_interval():
    y = np.arange(20.0)
    p = y + np.sin(y)
    m = scalar_metrics(y, p)
    assert m["pearson_r"] == pytest.approx(np.corrcoef(y, p)[0, 1])
    assert m["r_lower"] < m["pearson_r"] < m["r_upper"]


def test_binary_direction_not_optimized():
    y, p = np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])
    m = scalar_metrics(y, p, binary=True)
    assert m["auc"] == roc_auc_score(y, p) == 0
    assert m["balanced_accuracy"] == 0


def test_constants_explicit_and_chance_auc():
    m = scalar_metrics([0, 0, 1, 1], [0.4] * 4, binary=True)
    assert m["pearson_r"] is None
    assert m["auc"] == 0.5


def packet(group="mbti_only"):
    """Create a three-view author record without source text or identities."""
    return {
        "x": {
            "author_id": "x",
            "source_group": group,
            "view_means": {"raw__EI_cont": 0.5},
            "readouts": [
                {
                    "task": "EI_cont",
                    "arm": "raw",
                    "view": i,
                    "conditional_score": v,
                    "candidate_mass": 0.8,
                    "conditional_entropy": 0.5,
                    "greedy_first_token_is_candidate": True,
                }
                for i, v in enumerate([0.2, 0.5, 0.8])
            ],
        }
    }


def test_three_view_mean():
    p, rows = direct_prediction(packet(), ["x"], "EI_cont", "raw")
    assert p.tolist() == [0.5]
    assert len(rows) == 3


def test_missing_view_rejected():
    data = packet()
    data["x"]["readouts"].pop()
    with pytest.raises(ValueError, match="three"):
        direct_prediction(data, ["x"], "EI_cont", "raw")


def test_bridge_role_rejected():
    with pytest.raises(ValueError, match="bridge"):
        direct_prediction(packet("bridge"), ["x"], "EI_cont", "raw")
