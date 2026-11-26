#!/usr/bin/env python3
import itertools
import json
import sys

import pytest

from grain_core import event_response as er
from grain_core import response_profile as rp


def reference(frame="presented_opportunities"):
    return {
        "behavior": "give an example",
        "condition_rule": "preceding prompt",
        "unit": "single invitation",
        "frame": frame,
        "reader": "test-v1",
        "weights": {"a": 0.5, "b": 0.5},
        "contrast": ["a", "b"],
    }


def dataset(a, b=(), person="u"):
    return [
        {
            "dialogue": "d",
            "focal": person,
            "condition": c,
            "id": f"{c}{i}",
            "status": s,
            "response_id": f"r-{c}{i}" if s in rp.REPLIES else None,
        }
        for c, states in zip("ab", (a, b), strict=True)
        for i, s in enumerate(states)
    ]


def profile(a, b=(), ref=None):
    return rp.measure(ref or reference(), dataset(a, b))["cards"][0]


def test_same_average_opposite_response_and_exact_mix_accounting():
    yes, no = "support", "refute"
    a = dataset([yes] * 8 + [no] * 2, [yes] + [no] * 4, "u")
    b = dataset([yes] * 4 + [no], [yes] * 2 + [no] * 8, "v")
    r = rp.measure(reference(), a + b)
    u, v = r["cards"]
    assert u["standardized"] == v["standardized"] == [0.5, 0.5]
    assert u["natural"] == pytest.approx([0.6, 0.6])
    d = rp.compare(r, r, "u", "v")["decomposition"]
    assert d["recorded_condition_mix"] == pytest.approx(-0.2)
    assert d["within_condition_response"] == pytest.approx(0)
    rev = profile([no] * 4 + [yes], [yes] * 4 + [no])
    assert rev["standardized"] == [0.5, 0.5]
    assert rev["response_contrast"] == pytest.approx([-0.6, -0.6])


def test_unknown_bounds_match_all_feasible_completions():
    statuses = ["support", "refute", "no_reply", "unresolved", "reply_unknown"]
    a = profile(statuses)["conditions"]["a"]
    totals = []
    replies = []
    per = []
    for s, t in itertools.product(
        ("support", "refute"), ("support", "refute", "no_reply")
    ):
        c = statuses[:3] + [s, t]
        n = len(c) - c.count("no_reply")
        totals.append(c.count("support") / len(c))
        replies.append(n / len(c))
        per.append(c.count("support") / n)
    for key, vals in (
        ("behavior_given_reply", per),
        ("reading_bounds", totals),
        ("reply_rate", replies),
    ):
        assert a[key] == pytest.approx([min(vals), max(vals)])


def test_no_reply_is_not_zero_conditional_behavior_or_unknown_opportunity():
    c = profile(["no_reply", "opportunity_unconfirmed"])
    assert c["conditions"]["a"]["behavior_given_reply"] is None
    assert c["natural"] == [0.0, 0.0] and c["unsupported_weight"] == 0.5
    assert c["standardized"] == [0.0, 0.5] and c["unconfirmed_excluded"] == 1
    c = profile(["reply_unknown"])
    assert c["conditions"]["a"]["given_reply_denominator_may_be_zero"]
    assert c["natural"] == [0.0, 1.0]
    assert profile(["opportunity_unconfirmed"])["natural"] is None


def test_mixture_gap_reuses_same_unknown_and_no_forced_split():
    c = profile(["unresolved"], ["unresolved"])
    assert c["natural"] == c["standardized"] == [0, 1]
    assert c["mix_gap_natural_minus_standardized"] == [0.0, 0.0]
    r = rp.measure(
        reference(), dataset(["support"], person="u") + dataset([], ["refute"], "v")
    )
    assert rp.compare(r, r, "u", "v")["status"] == "unobserved_condition_response"
    with pytest.raises(ValueError, match="frozen reference"):
        rp.compare(r, {**r, "reference": {}}, "u", "v")


def test_participation_separate_from_expression():
    r = rp.measure(
        reference(),
        dataset(["no_reply", "support"], ["support", "refute"], "u")
        + dataset(["support", "support"], ["support", "support"], "v"),
    )
    d = rp.compare(r, r, "u", "v")["decomposition"]
    assert d["speaking_given_opportunity"] == d["expression_given_speaking"] == 0.25
    assert d["closure_error"] == 0.0
    r = rp.measure(
        reference("observed_replies"),
        dataset(["support"], ["refute"], "u") + dataset(["refute"], ["support"], "v"),
    )
    assert r["cards"][0]["denominator"] == "among_replies"
    assert r["cards"][0]["conditions"]["a"]["reply_rate"] is None
    assert (
        rp.compare(r, r, "u", "v")["decomposition"]["speaking_given_opportunity"]
        is None
    )
    with pytest.raises(ValueError, match="replies-only"):
        rp.measure(reference("observed_replies"), dataset(["reply_unknown"]))


def test_no_double_count_and_freeze_reference():
    events = dataset(["support"])
    for extra in (events[0], dict(events[0], id="other")):
        with pytest.raises(ValueError):
            rp.measure(reference(), events + [extra])
    ref = reference()
    output = rp.measure(ref, events)
    ref["weights"]["a"] = 2
    assert output["reference"]["weights"]["a"] == 0.5
    with pytest.raises(ValueError):
        rp.measure(ref, events)


def test_addressed_event_cli_legacy_and_new_card(tmp_path, monkeypatch, capsys):
    doc = {
        "language": "en",
        "dialogues": [
            {
                "id": k,
                "condition": k,
                "turns": [
                    {"speaker": "p", "text": "It's blue, not red."},
                    {"speaker": "u", "text": "I'll use blue."},
                ],
            }
            for k in "ab"
        ],
    }
    probes = [
        {
            "dialogue": k,
            "focal": "u",
            "trigger": 0,
            "cutoff": cut,
            "old": "red",
            "new": "blue",
        }
        for k, cut in zip("ab", (1, 0), strict=True)
    ]
    f = tmp_path / "input.json"
    f.write_text(
        json.dumps(
            {
                "doc": doc,
                "probes": probes,
                "weights": reference()["weights"],
                "contrast": ["a", "b"],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["prog", str(f)])
    er.main()
    old = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(sys, "argv", ["prog", str(f), "--response-profile"])
    er.main()
    new = json.loads(capsys.readouterr().out)
    card = new.pop("response_profile")["cards"][0]
    assert old == new and card["natural"] == [0.5, 1.0]
    assert card["conditions"]["b"]["counts"] == {"reply_unknown": 1}


def test_arithmetic_roundtrip_cli(tmp_path, monkeypatch):
    f = tmp_path / "input.json"
    o = tmp_path / "result.json"
    f.write_text(
        json.dumps(
            {
                "reference": reference(),
                "events": dataset(["refute"]) + dataset([], ["support"], "v"),
                "comparisons": [["u", "v"]],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["prog", str(f), "--out", str(o)])
    rp.main()
    out = json.loads(o.read_text())
    assert out["comparisons"][0]["natural_change"] == [1.0, 1.0]
    with pytest.raises(FileExistsError):
        rp.main()


def test_jackknife_uses_independent_clusters_and_requires_coverage():
    rows = dataset(["support", "refute", "refute"])
    for i, r in enumerate(rows):
        r["partner"] = str(i)
    c = rp.measure(reference(), rows, sampling_cluster="partner")["cards"][0]
    se = c["sampling_precision"]["delete_one_cluster_se"]
    assert se["natural"] == pytest.approx(1 / 3) and se["standardized"] is None
    for r in rows:
        r["partner"] = "same"
    c = rp.measure(reference(), rows, sampling_cluster="partner")["cards"][0]
    assert c["sampling_precision"]["clusters"] == 1
    assert c["sampling_precision"]["delete_one_cluster_se"]["natural"] is None
