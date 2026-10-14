#!/usr/bin/env python3
"""Generic ledger tests; syntax stubbed so no neural parse runs."""

import grain_research.dialogue_ledger_generic as generic


def frame(channel="past_action_report", actor="respondent", predicate="ask", question=True, conditions=None, sign=1):
    return {
        "channel": channel, "actor": actor, "predicate": predicate, "question": question,
        "conditions": conditions or [], "sign": sign, "source_scoped": False,
        "ambiguous_negation": False, "target_relevant": False, "start": 0, "end": 3,
    }


def test_generic_question_binds_any_predicate(monkeypatch):
    monkeypatch.setattr(generic, "frames", lambda *_: [frame(predicate="travel", conditions=[{"start": 0, "end": 5, "text": "when tired"}])])
    turn = {"id": "p:t0", "speaker": "partner", "text": "When tired, did you travel?"}
    proposition, status, account = generic.generic_question(turn, [])
    assert status == "question_open"
    assert proposition["query"] == "past_action_report:travel"
    assert proposition["condition"] == "when tired" and proposition["condition_addressed"]
    assert account["eligible_frames"] == 1


def test_generic_question_reasons(monkeypatch):
    turn = {"id": "p:t0", "speaker": "partner", "text": "x?"}
    monkeypatch.setattr(generic, "frames", lambda *_: [])
    assert generic.generic_question(turn, [])[1] == "no_question_frame"
    monkeypatch.setattr(generic, "frames", lambda *_: [frame(actor="unresolved")])
    assert generic.generic_question(turn, [])[1] == "question_not_bindable"
    monkeypatch.setattr(generic, "frames", lambda *_: [frame(), frame(predicate="go")])
    assert generic.generic_question(turn, [])[1] == "multiple_bindable_questions"


def test_polar_prefix_is_candidate_not_score():
    kind, sign, _ = generic.classify_reply("Yes, I went there last week and it was fine.")
    assert kind == "polar_prefix_with_content" and sign == 1
    assert generic.classify_reply("Nope, never.")[1] == -1
    assert generic.classify_reply("Yes, I did.")[0] == "answer"
    assert generic.classify_reply("Probably not worth it.")[0] == "unresolved"


def test_read_dialogue_generic_scores_and_reasons(monkeypatch):
    monkeypatch.setattr(generic, "frames", lambda *_: [frame(predicate="travel")])
    turns = [
        {"id": "d:t0", "speaker": "partner", "text": "Did you travel?"},
        {"id": "d:t1", "speaker": "respondent", "text": "Yes, I did."},
        {"id": "d:t2", "speaker": "respondent", "text": "Yes, I did it because the weather was good."},
        {"id": "d:t3", "speaker": "respondent", "text": "I withdraw my previous answer."},
    ]
    result = generic.read_dialogue_generic(turns, {t["text"]: [] for t in turns})
    statuses = [op["status"] for op in result["operations"]]
    assert statuses == ["question_open", "new", "polar_prefix_with_content", "retract"]
    assert result["snapshots"][1]["scores"]["past_action_report:travel:None"]["score"] == 1.0
    assert result["snapshots"][3]["scores"] == {}


def test_replay_matches_frozen_account_on_conditioned_history():
    from grain_research.evidence_history import active_records

    history = [
        {"record_id": "a", "kind": "new", "event_id": "e1", "block_id": "b", "condition": -1, "value": 1.0},
        {"record_id": "b", "kind": "alias", "parent_id": "a"},
        {"record_id": "c", "kind": "revise", "event_id": "e1", "block_id": "b", "condition": -1, "value": -1.0, "parent_id": "a"},
        {"record_id": "d", "kind": "new", "event_id": "e2", "block_id": "b", "condition": 1, "value": 1.0},
        {"record_id": "e", "kind": "retract", "parent_id": "d"},
    ]
    frozen = sorted(r["record_id"] for r in active_records(history)["records"])
    ours = sorted(r["record_id"] for r in generic.replay_versions(history)["records"])
    assert frozen == ours == ["c"]


def test_polar_binding_records_with_elaboration_flag(monkeypatch):
    monkeypatch.setattr(generic, "frames", lambda text, *_: [frame(predicate="travel")] if "?" in text else [
        {**frame(question=False), "local_negation": "not" in text}
    ])
    turns = [
        {"id": "d:t0", "speaker": "partner", "text": "Did you travel?"},
        {"id": "d:t1", "speaker": "respondent", "text": "Yes, but honestly I did not enjoy it."},
    ]
    off = generic.read_dialogue_generic(turns, {t["text"]: [] for t in turns})
    assert off["operations"][1]["status"] == "polar_prefix_with_content"
    on = generic.read_dialogue_generic(turns, {t["text"]: [] for t in turns}, polar_binding=True)
    assert on["operations"][1]["status"] == "new_polar_with_elaboration_negation_flag"
    record = on["history"][0]
    assert record["confidence"] == "polar_with_elaboration" and record["value"] == 1.0
    assert on["snapshots"][1]["scores"]["past_action_report:travel:None"]["score"] == 1.0
