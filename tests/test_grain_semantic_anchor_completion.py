#!/usr/bin/env python3
"""Historical metadata compatibility, separate from the frozen scientific code."""

from scripts.complete_grain_semantic_anchor_hm147 import control_indices


def test_historical_none_axis_is_a_control_not_a_target():
    rows = [
        {
            "axis": "none",
            "pole": 0,
            "condition": "neutral",
            "situation": "advice",
            "seed": 11,
        },
        {
            "axis": "none",
            "pole": 0,
            "condition": "nonpersona_spelling",
            "situation": "advice",
            "seed": 11,
        },
        {
            "axis": "extraversion",
            "pole": 1,
            "condition": "extraversion_high",
            "situation": "advice",
            "seed": 11,
        },
    ]
    assert control_indices(rows) == {
        ("neutral", "advice", 11): 0,
        ("nonpersona_spelling", "advice", 11): 1,
    }
