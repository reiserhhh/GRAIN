#!/usr/bin/env python3
"""Verify that partial captures cannot masquerade as full response laws."""

from scripts.describe_grain_writer_history_partial_hm165 import complete_tables


def test_only_complete_actual_tables_are_used_and_missing_is_retained():
    requests, captures = [], {}
    for pair in ("complete", "missing"):
        for p in (-1, 0, 1):
            for h in (-1, 0, 1):
                index = len(requests)
                requests.append(
                    {
                        "pair_id": pair,
                        "opportunity": "social",
                        "order": 0,
                        "policy": p,
                        "history": h,
                        "unit_index": index,
                    }
                )
                if pair == "complete" or (p, h) != (1, 1):
                    captures[index] = {"ab_logits": [1, 2], "log_normalizer": 4}
    full, missing = complete_tables(requests, captures)
    assert len(full) == 1
    assert missing == [
        {
            "pair_id": "missing",
            "opportunity": "social",
            "order": 0,
            "captured_corners": 8,
        }
    ]


def test_empty_capture_is_not_an_observed_zero():
    request = {
        "pair_id": "p",
        "opportunity": "a",
        "order": 0,
        "policy": 0,
        "history": 0,
        "unit_index": 0,
    }
    full, missing = complete_tables([request], {})
    assert not full
    assert missing[0]["captured_corners"] == 0
