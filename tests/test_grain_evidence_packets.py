#!/usr/bin/env python3
"""Evidence manipulations must cross intended variables without outcome-based selection."""

from collections import Counter

import pandas as pd
import pytest

from grain_research.vector_mechanisms.evidence_packets import (
    binary_query_factorial,
    natural_evidence_edits,
    temporal_worlds,
)


def test_binary_factorial_matches_polarity_and_alphabets():
    q = binary_query_factorial()
    assert len(q) == 4
    assert Counter(v["framing"] for v in q.values()) == {
        "big5_binary": 2,
        "mbti_binary": 2,
    }
    assert {tuple(v["values"]) for v in q.values()} == {(0, 1)}
    for coding in ("digits", "letters"):
        assert (
            q[f"big5_binary_{coding}"]["labels"] == q[f"mbti_binary_{coding}"]["labels"]
        )


def test_natural_edits_preserve_body_and_nonselected_excerpts():
    original = [
        "A fact about the weather.",
        "I feel quiet in crowds.",
        "An unrelated note.",
    ]
    edits = natural_evidence_edits(original)
    assert edits["selected_excerpt"] == 1 and len(edits["variants"]) == 8
    for name, texts in edits["variants"].items():
        if name != "remove":
            assert (
                texts[0] == original[0]
                and texts[2] == original[2]
                and original[1] in texts[1]
            )
    assert edits["variants"]["remove"] == [original[0], original[2]]
    assert original[1] == "I feel quiet in crowds."
    assert natural_evidence_edits(["Just a note.", "Another note."]) is None


def test_time_assignment_is_independent_of_display_order():
    worlds = temporal_worlds()
    assert len(worlds) == 24 and len({w["world_id"] for w in worlds}) == 24
    for i in range(6):
        for neg in (False, True):
            pair = [
                w
                for w in worlds
                if w["scene_id"] == f"t{i:02d}" and w["negative_now"] == neg
            ]
            assert pair[0]["texts"] == list(reversed(pair[1]["texts"]))
            assert all(not w["event_labels_are_human_personality_truth"] for w in pair)


def test_factorial_responses_pair_exact_worlds(monkeypatch):
    import scripts.analyze_grain_evidence_packets as analysis

    monkeypatch.setattr(
        analysis,
        "cluster_r_interval",
        lambda x, y, groups: {"r": 1.0, "r_lower": None, "r_upper": None},
    )
    rows = []
    for scene in range(2):
        for actor, a in (("self", 0), ("other", 1)):
            for opportunity, b in (("voluntary", 0), ("assigned", 1)):
                for renderer in (0, 1):
                    for readout, multiplier in (("native", 1), ("R_new", 2)):
                        rows.append(
                            {
                                "query": "q",
                                "readout": readout,
                                "scene_id": str(scene),
                                "split": "held_scene",
                                "actor": actor,
                                "opportunity": opportunity,
                                "realization": renderer,
                                "score": multiplier
                                * (scene + 0.2 * a + 0.3 * b + 0.04 * renderer),
                            }
                        )
    f = pd.DataFrame(rows)
    r = analysis.response_rows(f, "identity")
    joined, metrics = analysis.compare_responses(r)
    assert len(joined) == 24 and len(metrics) == 3
    assert (joined.score_delta - 2 * joined.native_delta).abs().max() < 1e-12
    with pytest.raises(ValueError, match="complete"):
        analysis.response_rows(f.iloc[:-1], "identity")
