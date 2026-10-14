#!/usr/bin/env python3
"""Tests for the new-text dialogue ledger entry; no neural parse is invoked."""

import json

import pytest

import grain_research.dialogue_claim_ledger as ledger
from scripts import score_grain_dialogue_ledger_input as entry
from scripts.run_grain_dialogue_claim_ledger import public_bank


def test_bounded_mean_range_identity():
    assert entry.bounded_mean_range(0.0, 0, 0) is None
    assert entry.bounded_mean_range(0.0, 0, 3) == [-1.0, 1.0]
    low, high = entry.bounded_mean_range(2.0, 4, 2)
    assert low == pytest.approx(0.0) and high == pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        entry.bounded_mean_range(0.0, -1, 0)


def install_stub(monkeypatch):
    def question(turn, _parsed, _calibration):
        return {
            "question_id": turn["id"],
            "actor": "respondent",
            "condition": -1,
            "mode": "past_action_report",
            "question_span": ledger.address(turn),
        }, "question_open"

    monkeypatch.setattr(ledger, "question_proposition", question)
    monkeypatch.setattr(ledger, "speech_observations", lambda *_: [])
    monkeypatch.setattr(entry, "frames", lambda *_: [])


def test_computability_account_matches_authored_ledger(monkeypatch):
    install_stub(monkeypatch)
    row = public_bank()[0]
    parses = {t["text"]: [] for t in row["turns"]}
    _, _, turns, targets = entry.account_dialogue(row, parses, {})
    assert [t["operation"] for t in turns] == row["expected_operations"]
    by_target = {t["target"]: t for t in targets}
    cell = by_target["past_action_report:-1"]
    assert cell["computable"] and cell["score"] == row["expected_scores"][-1]
    assert cell["active_claims"] == 1
    assert not by_target["intention:1"]["computable"]
    assert by_target["intention:1"]["score"] is None
    assert cell["known_condition_range"] == [-1.0, -1.0]


def test_bare_answer_without_question_is_unassigned_not_scored(monkeypatch):
    install_stub(monkeypatch)
    row = {
        "id": "n0",
        "turns": [{"id": "n0:t0", "speaker": "respondent", "text": "Yes."}],
    }
    _, _, turns, targets = entry.account_dialogue(row, {"Yes.": []}, {})
    assert turns[0]["operation"] == "no_unique_question"
    assert turns[0]["polarity"] == 1
    for cell in targets:
        assert not cell["computable"] and cell["score"] is None
        assert cell["unassigned_polarity_turns"] == 1
        assert cell["known_condition_range"] is None
        assert cell["unknown_condition_range"] == [-1.0, 1.0]


def test_input_and_cache_loaders(tmp_path):
    bank = public_bank()[:2]
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(bank))
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps({"dialogues": bank}))
    assert [r["id"] for r in entry.load_dialogues(list_path)] == ["d00", "d01"]
    assert [r["id"] for r in entry.load_dialogues(dict_path)] == ["d00", "d01"]
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps([bank[0], bank[0]]))
    with pytest.raises(ValueError):
        entry.load_dialogues(dup)
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"Yes.": [["x"]]}))
    b = tmp_path / "b.json"
    b.write_text(json.dumps([{"text": "No.", "sentences": [["y"]]}]))
    cache = entry.load_parse_cache([a, b])
    assert cache == {"Yes.": [["x"]], "No.": [["y"]]}
