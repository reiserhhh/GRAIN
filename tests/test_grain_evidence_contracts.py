#!/usr/bin/env python3
"""Cache, annotation and confusion-axis regressions from independent review."""

import hashlib
import json

import pandas as pd
import pytest

from grain_research.evidence_contracts import (
    literal_confusion,
    load_model_annotations,
    validate_evidence_rows,
)


def test_threshold_flag_and_item_key_are_checked():
    sentences = [
        {"author_index": 0, "comment_index": 0, "comment_id": "c", "start": 0, "end": 3}
    ]
    items = [{"id": "q", "key": -1, "facet": "E1", "trait": "Extraversion"}]
    e = pd.DataFrame(
        [
            {
                **sentences[0],
                **{k: v for k, v in items[0].items() if k != "id"},
                "item_id": "q",
                "similarity": 0.7,
                "accepted": True,
            }
        ]
    )
    validate_evidence_rows(e, sentences, items, 0.6)
    e.loc[0, "accepted"] = False
    with pytest.raises(ValueError, match="threshold"):
        validate_evidence_rows(e, sentences, items, 0.6)
    e.loc[0, "accepted"] = True
    e.loc[0, "key"] = 1
    with pytest.raises(ValueError, match="key"):
        validate_evidence_rows(e, sentences, items, 0.6)


def test_annotations_are_hash_bound_and_enum_checked(tmp_path):
    d = {
        "rater_kind": "model",
        "rater_model": "gpt-5.6-sol",
        "scope": "independent_blinded_model_annotation_not_human_gold",
        "rows": [
            {
                "audit_id": "p",
                "self_attribution": "self",
                "scope": "general",
                "relation": "entails",
                "simpa_category": 1,
                "rationale": "synthetic",
            }
        ],
    }
    p = tmp_path / "a.json"
    p.write_text(json.dumps(d))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    _, frame = load_model_annotations(p, ["p"], sha)
    assert len(frame) == 1
    with pytest.raises(ValueError, match="hash"):
        load_model_annotations(p, ["p"], "bad")
    d["rows"][0]["self_attribution"] = "almost_self"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="category"):
        load_model_annotations(p, ["p"], hashlib.sha256(p.read_bytes()).hexdigest())


def test_confusion_expected_rows_match_reported_counts():
    f = pd.DataFrame(
        {
            "literal_expected": ["entailment"] * 2
            + ["contradiction"] * 2
            + ["neutral"] * 50,
            "nli": ["entailment"] * 2
            + ["contradiction"] * 2
            + ["neutral"] * 34
            + ["contradiction"] * 13
            + ["entailment"] * 3,
        }
    )
    result = literal_confusion(f)
    assert result["row_axis"] == "independent_model_expected"
    assert result["counts"]["neutral"]["contradiction"] == 13
    assert result["counts"]["contradiction"]["neutral"] == 0
    assert int(f.literal_expected.eq(f.nli).sum()) == 38
    assert int(f.literal_expected.eq("neutral").sum()) == 50
