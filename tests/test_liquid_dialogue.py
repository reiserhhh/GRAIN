#!/usr/bin/env python3
from copy import deepcopy

import numpy as np
import pytest
from grain_core.liquid_dialogue import attribute, prepare


def example():
    return {
        "language": "en",
        "clock": "opportunity",
        "dialogues": [
            {
                "id": "d",
                "condition": "one",
                "carrier": "chat",
                "view": "A",
                "turns": [
                    {"speaker": w, "text": t}
                    for w, t in [
                        ("a", "first turn"),
                        ("b", "second turn"),
                        ("a", "third turn"),
                    ]
                ],
            }
        ],
    }


def test_separates_partner_from_same_turn_slices_and_binds_units():
    doc = example()
    p, layout = prepare(doc)
    assert [len(e["events"]) for e in p["episodes"]] == [3, 2, 1]
    by = {}
    for e, turns, tags in zip(
        p["episodes"],
        [[0, 0, 1, 2], [0, 0, 1], [0]],
        [["i", "j", "k", "l"], ["i", "j", "l"], ["k"]],
        strict=True,
    ):
        n = len(turns)
        d = np.triu(np.ones((n, n)))[:, :, None]
        by[e["id"]] = {
            **e,
            "slots": [[v, turns[:i].count(v)] for i, v in enumerate(turns)],
            "unit_digests": tags,
            "liquid": [[0.0]] * n,
            "missing_effect": d.tolist(),
            "nine_fields": {
                "names": ["one"],
                "source_deletion_delta": (d * 2).tolist(),
            },
        }
    result = {"episodes": by, "node_names": ["a"], "reference_digest": "x"}
    score = attribute(doc, layout, result)["dialogues"]["d"]
    a = score["speaker_paths"]["a"]
    assert a["joint_slice_indices"] == [0, 1, 3]
    assert a["prior_own_turn_single_omission_sum"] == [[0], [0], [2]]
    assert a["prior_partner_turn_single_omission_sum"] == [[0], [0], [1]]
    assert a["nine_fields"]["prior_partner_turn_single_omission_sum"] == [[0], [0], [2]]
    assert "author" not in score["joint_trajectory"]
    by["d0:speaker1"]["unit_digests"][0] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        attribute(doc, layout, result)


def test_refuses_ambiguous_people_and_duplicate_episodes():
    doc = example()
    doc["dialogues"].append(deepcopy(doc["dialogues"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        prepare(doc)
    doc = example()
    doc["dialogues"][0]["turns"][1]["speaker"] = "a"
    with pytest.raises(ValueError, match="two named"):
        prepare(doc)
