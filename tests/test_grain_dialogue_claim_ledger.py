#!/usr/bin/env python3
"""Unit tests for version identities; dependency grammar tested in public run."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import grain_research.dialogue_claim_ledger as ledger
from scripts.run_grain_dialogue_claim_ledger import public_bank


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes, I did.", ("answer", 1, True)),
        ("I don't.", ("answer", -1, True)),
        ("No.", ("answer", -1, False)),
        ("Let me correct my previous answer: No, I do not.", ("revise", -1, True)),
        ("I am repeating my previous answer: Yes, I do.", ("alias", 1, True)),
        ("I take back my previous answer.", ("retract", None, False)),
        ("I intended to ask, but I never did.", ("unresolved", None, False)),
    ],
)
def test_explicit_operations(text, expected):
    assert ledger.answer_operation(text) == expected


def test_condition_frozen_transform():
    """Verify the no-fit adapter matches source sklearn TF-IDF exactly."""
    texts = ["When the deadline approaches", "When instructions are missing"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    source = vectorizer.fit_transform(texts).toarray()
    calibration = {
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_,
        "centroids": source,
    }
    for text in texts + ["when you face a deadline", "when you lack instructions"]:
        expected = (
            -1
            if (source @ vectorizer.transform([text]).toarray()[0]).argmax() == 0
            else 1
        )
        assert ledger.frozen_condition(text, calibration) == expected
    assert ledger.frozen_condition("unmapped vocabulary", calibration) is None


def install_question(monkeypatch):
    """Stub syntax only so version tests do not invoke neural inference."""

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


def test_versioned_dialogue(monkeypatch):
    install_question(monkeypatch)
    row = public_bank()[0]
    result = ledger.read_dialogue(
        row["turns"], {t["text"]: [] for t in row["turns"]}, {}
    )
    assert [r["status"] for r in result["operations"]] == row["expected_operations"]
    assert [
        s["scores"]["past_action_report:-1"]["score"] for s in result["snapshots"]
    ] == row["expected_scores"]
    assert (
        result["snapshots"][2]["scores"]["past_action_report:-1"]["active_claims"] == 1
    )
    assert (
        result["snapshots"][5]["scores"]["past_action_report:-1"]["active_claims"] == 2
    )
    assert (
        result["snapshots"][6]["scores"]["past_action_report:-1"]["active_claims"] == 1
    )
    assert len(result["snapshots"][3]["stale_derived"]) == 1
    assert all(
        not r["physical_event_identity_verified"]
        for r in result["history"]
        if "value" in r
    )


def test_missing_is_not_negative():
    snapshot = ledger.numeric_snapshot([])
    assert all(
        s["score"] is None and s["active_claims"] == 0
        for s in snapshot["scores"].values()
    )


def test_repeat_conflict_and_orphan_repair(monkeypatch):
    install_question(monkeypatch)
    for row in public_bank()[8:11]:
        if row["challenge"] == "negative_question":
            continue
        result = ledger.read_dialogue(
            row["turns"], {t["text"]: [] for t in row["turns"]}, {}
        )
        assert [r["status"] for r in result["operations"]] == row["expected_operations"]


def test_role_and_duplicate_validation():
    with pytest.raises(ValueError, match="two-party"):
        ledger.read_dialogue(
            [{"id": "x", "speaker": "unknown", "text": "Yes."}], {"Yes.": []}, {}
        )
    t = {"id": "x", "speaker": "respondent", "text": "Yes."}
    with pytest.raises(ValueError, match="unique"):
        ledger.read_dialogue([t, t], {"Yes.": []}, {})


def test_source_and_proposed_actor(monkeypatch):
    frame = {
        "id": "1",
        "channel": "current_request_candidate",
        "source_scoped": False,
        "predicate": "help",
        "start": 7,
        "end": 11,
    }
    monkeypatch.setattr(ledger, "frames", lambda *_: [frame])
    turn = {"id": "x", "speaker": "respondent", "text": "Please help me."}
    observed = ledger.speech_observations(turn, [], "partner")[0]
    assert observed["speech_source"] == "respondent"
    assert observed["proposed_action_actor"] == "partner"
    assert observed["proposed_action_enacted"] is None
    assert observed["span"]["quote"] == "help"
    frame["source_scoped"] = True
    assert ledger.speech_observations(turn, [], "partner") == []


def test_bank_budget_and_expected_separation():
    bank = public_bank()
    assert len(bank) == 14
    assert len({t["text"] for r in bank for t in r["turns"]}) <= 100
    assert all(set(t) == {"id", "speaker", "text"} for r in bank for t in r["turns"])
    assert np.isfinite(
        [v for r in bank[:8] for v in r["expected_scores"] if v is not None]
    ).all()
