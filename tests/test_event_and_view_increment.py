#!/usr/bin/env python3
import copy

import numpy as np
import pytest
from grain_core.dual_view_increment import assess, compare, moment_screen
from grain_core.event_response import (
    conditional_card,
    inspect_event,
    joined_opportunity,
    later_check,
)
from grain_core.liquid_instrument import digest


def example(reply="I'll use blue."):
    doc = {
        "language": "en",
        "dialogues": [
            {
                "id": "d",
                "condition": "c",
                "turns": [
                    {"speaker": "b", "text": "It's blue, not red."},
                    {"speaker": "a", "text": reply},
                    {"speaker": "b", "text": "later information"},
                ],
            }
        ],
    }
    probe = {
        "dialogue": "d",
        "trigger": 0,
        "cutoff": 1,
        "focal": "a",
        "old": "red",
        "new": "blue",
    }
    return doc, probe


@pytest.mark.parametrize(
    ("reply", "status"),
    [
        ("I'll use blue.", "support"),
        ("I choose BLUE!", "support"),
        ("I'll use red.", "refute"),
        ("I'll use blue?", "insufficient"),
        ('She said "I choose blue".', "insufficient"),
        ("I won't use red.", "insufficient"),
        ("Okay.", "insufficient"),
    ],
)
def test_addressed_vote(reply, status):
    d, p = example(reply)
    e = inspect_event(d, p)
    assert e["status"] == status and e["response"]["quote"] == reply
    assert e["response"]["timestamp"] is None
    assert inspect_event(d, dict(p, cutoff=0))["response"] is None
    assert inspect_event(d, dict(p, focal="b"))["opportunity"] == "trigger_unconfirmed"
    d["dialogues"][0]["turns"][2]["text"] = "I choose red."
    assert inspect_event(d, p) == e


def test_conflict_and_conditional_identification_bounds():
    d, p = example()
    e = inspect_event(d, p)
    vote = {
        "version": "frozen",
        "kind": "coded_response",
        "vote": "refute",
        "actor": "a",
        "prefix_id": e["prefix_id"],
    }
    assert (
        inspect_event(d, dict(p, semantic=dict(vote, kind="candidate")))["status"]
        == "support"
    )
    conflict = inspect_event(d, dict(p, semantic=vote))
    assert conflict["status"] == "insufficient" and "conflict" in conflict["reason"]
    with pytest.raises(ValueError):
        inspect_event(d, dict(p, semantic=dict(vote, actor="b")))
    with pytest.raises(ValueError):
        inspect_event(d, dict(p, semantic=dict(vote, prefix_id="later")))
    with pytest.raises(ValueError, match="shared action"):
        conditional_card([e, conflict], "a", {"c": 1})
    censor = inspect_event(d, dict(p, cutoff=0))
    card = conditional_card([e, censor], "a", {"c": 0.25, "other": 0.75})
    assert card["reference_rate_bounds"] == [0.25, 1.0]
    assert card["unconfirmed_or_no_observed_reply"] == {"reply_unobserved": 1}
    assert (
        joined_opportunity([conflict, conflict], [[1, 0], [0, 1]])["status"] == "refute"
    )
    assert joined_opportunity([e], [[0]])["reason"] == "joint_conflict_review"
    assert (
        later_check(
            conflict,
            {
                "dialogue": "d",
                "actor": "b",
                "id": "1",
                "choice": "blue",
                "source": "log",
                "after_turn": 2,
            },
        )["focal_choice_of_premise"]
        == "unresolved"
    )


def test_optional_readers_provenance():
    d, p = example("I'll use red.")
    sidecar = {
        "input_digest": digest(d),
        "kind": "dyad_reader_not_personal_factor_scores",
        "reference_digest": "frozen",
        "clock": "opportunity",
        "dialogues": {
            "d": {
                "speaker_at_slice": ["b", "a", "b"],
                "joint_trajectory": {
                    "slots": [[0, 0], [1, 0], [2, 0]],
                    "liquid": [[1], [30], [200]],
                },
            }
        },
    }
    e = inspect_event(d, p, sidecar)
    assert e["status"] == "refute" and e["liquid"]["readout"] == [[30]]
    with pytest.raises(ValueError):
        inspect_event(d, p, dict(sidecar, input_digest="bad"))
    foreign = copy.deepcopy(d)
    foreign["language"] = "ja"
    e = inspect_event(foreign, p)
    corrected = dict(
        p,
        correction={
            "prefix_id": e["trigger_prefix_id"],
            "confirmed": True,
            "version": "coding",
        },
    )
    assert inspect_event(foreign, corrected)["status"] == "insufficient"


def test_matched_calibration_and_drop_to_direct():
    options = {"dates1": ["a", "b"], "dates2": ["a", "b"]}
    assert assess([0, 1], [1, 2], **options)["status"] == "direct_only"
    for dates in (["a", "c"], ["a", "a"], None):
        with pytest.raises(ValueError):
            assess([0, 1], [1, 2], **(options | {"dates2": dates}))
    p = [[0, 2], [1, 1], [2, 0]]
    assert moment_screen(p)["common_sample_covariance"] == -1
    assert assess([0, 1], [1, 2], previous_levels=p)["full"] is None
    with pytest.raises(ValueError):
        assess([0, np.nan], [0, 2])


def test_witnesses_use_same_history():
    kw = {
        "view1": [0, 1],
        "view2": [0, 1],
        "half_widths": [2, 3],
        "rates": [0, 0.8, -0.8],
        "budgets": [2, 0.2, 0.2],
        "c_bound": 10,
        "calibration_id": "known",
    }
    row = compare(**kw)
    assert row["simple_lag"] == pytest.approx([-0.4, 2])
    assert row["full"] == pytest.approx([0.8, 1.2])
    assert row["vs_simple_lag"] == "changes_threshold_decision"
    assert compare(**kw, threshold=4)["vs_simple_lag"] == "narrows_only"
    exact = dict(
        kw,
        view1=[2 - 1 / 0.9, 0],
        view2=[4.5, 0],
        half_widths=[8, 8],
        rates=[0.5, 0.9, -0.4],
        budgets=[0, 0, 0],
    )
    assert compare(**exact)["vs_simple_lag"] == "simple_lag_sufficient"
    assert compare(**(exact | {"view2": [4.5, 2]}))["status"] == "incompatible_private"


def test_random_feasible_truth_nested_including_signed_rates():
    rng = np.random.default_rng(431)
    for _ in range(25):
        r = rng.uniform(-0.9, 0.9, size=3)
        paths = [rng.uniform(-1, 1, size=3)]
        for _t in range(4):
            paths.append(r * paths[-1] + rng.uniform(-0.19, 0.19, size=3))
        c, p, q = np.array(paths).T
        row = compare(
            c + p,
            c + q,
            abs((p + q) / 2) + 0.1,
            r,
            [0.2] * 3,
            10,
            calibration_id="known",
        )
        lo, hi = row["full"]
        assert lo - 1e-8 <= c[-1] <= hi + 1e-8
        assert row["simple_lag"][0] - 1e-8 <= lo and hi <= row["simple_lag"][1] + 1e-8
