#!/usr/bin/env python3
"""Event addresses and sampling units cannot be replaced by author-only joins."""

import copy

import numpy as np
import pandas as pd
import pytest

from grain_research.gptpro_foundation_v3.f16_event_addresses import (
    event_address_frame,
    text_hash,
    verify_event_matches,
    x_event_tables,
)


def material_case():
    comments = [
        {
            "author_id": "SIM",
            "comment_id": f"e{i}",
            "thread_id": f"t{i}",
            "body": body,
            "timestamp": 1704067200 + i * 100,
            "subreddit": "A",
        }
        for i, body in enumerate(("first", "second"))
    ]
    doc = "first\n\nsecond"
    material = {
        "bundles": [
            {"author_id": "SIM", "source_group": "big5_only", "views": [comments]}
        ],
        "documents": [doc],
    }
    names = {"ids": ["SIM"], "document_hashes": [text_hash(doc)]}
    addresses = pd.DataFrame(
        {"author_index": [0, 0], "unit": [0, 1], "start": [0, 7], "end": [5, 13]}
    )
    return material, names, addresses, np.array([0, 0]), np.array(["A", "A"])


def test_exact_event_mapping_and_same_author_wrong_span_rejected():
    values = material_case()
    out = event_address_frame(*values)
    assert out.event_key.tolist() == ["e0", "e1"]
    bad = copy.deepcopy(values)
    bad[2].loc[0, "end"] += 1
    with pytest.raises(ValueError, match="span"):
        event_address_frame(*bad)
    bad = copy.deepcopy(values)
    bad[0]["bundles"][0]["views"][0][0]["body"] = "FIRST"
    with pytest.raises(ValueError, match="aggregation"):
        event_address_frame(*bad)


def test_raw_event_join_retains_duplicates_and_rejects_conflicts():
    mapped = event_address_frame(*material_case())
    found = pd.concat([mapped, mapped.iloc[:1]], ignore_index=True)
    assert verify_event_matches(mapped, found)["repeat_source_rows"] == 1
    with pytest.raises(ValueError, match="missing or extraneous"):
        verify_event_matches(mapped, found.iloc[:1])
    found.loc[2, "timestamp_unix_seconds"] += 1
    with pytest.raises(ValueError, match="conflicting duplicate"):
        verify_event_matches(mapped, found)


def test_x_event_collection_separation_and_partial_context():
    frame = pd.DataFrame(
        {
            "post_id": ["p", "p", "q"],
            "account_id": ["a"] * 3,
            "timestamp": ["2024-01-01T00:00:00Z"] * 3,
            "lang": ["en"] * 3,
            "symbol": ["A", "B", ""],
            "query_group": ["Q"] * 3,
        }
    )
    events, tags = x_event_tables(frame)
    assert len(events) == 2 and len(tags) == 3
    assert events.collection_rows.tolist() == [2, 1]
    assert tags.complete_condition.tolist() == [True, True, False]
    assert tags.partial_reason.iloc[-1] == "symbol"
    assert pd.isna(tags.condition_key.iloc[-1])
    bad = frame.copy()
    bad.loc[1, "account_id"] = "b"
    with pytest.raises(ValueError, match="conflicting post"):
        x_event_tables(bad)
    bad = frame.copy()
    bad.loc[1, "timestamp"] = "2024-01-01T00:00:00"
    with pytest.raises(ValueError, match="timezone"):
        x_event_tables(bad)
