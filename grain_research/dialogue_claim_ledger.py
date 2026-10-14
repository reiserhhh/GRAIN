#!/usr/bin/env python3
"""Bounded raw-dialogue claim updates with separate speech/action ownership.

Explicit reference phrases license version operations. Predicate ellipsis is
resolved only against a unique current question or explicitly referenced claim.
The ledger contains claims about events, not independently verified actions.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from copy import deepcopy

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from grain_research.evidence_history import active_records
from grain_research.proposition_response import frames

REPAIR = re.compile(
    r"^\s*(?:I need to correct|Let me correct) my previous answer:\s*", re.IGNORECASE
)
REPEAT = re.compile(
    r"^\s*(?:To repeat|I am repeating) my previous answer:\s*", re.IGNORECASE
)
RETRACT = re.compile(
    r"^\s*I (?:withdraw|take back) my previous answer[.!]?\s*$", re.IGNORECASE
)
ANSWER = re.compile(
    r"^\s*(?:(?:Yes|No)[, ]+)?I (do|did|don't|didn't|do not|did not)[.!]?\s*$",
    re.IGNORECASE,
)
BARE = re.compile(r"^\s*(Yes|No)[.!]?\s*$", re.IGNORECASE)


def address(turn, start=0, end=None):
    """Bind a Unicode span to its exact observed turn."""
    text = turn["text"]
    end = len(text) if end is None else end
    return {
        "turn_id": turn["id"],
        "speaker": turn["speaker"],
        "start": start,
        "end": end,
        "quote": text[start:end],
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def frozen_condition(text, calibration):
    """Apply source-fitted TF-IDF condition prototypes without fitting anything."""
    analyzer = TfidfVectorizer(ngram_range=(1, 2)).build_analyzer()
    counts = Counter(analyzer(text))
    vocab = calibration["vocabulary"]
    vector = np.zeros(len(vocab))
    for term, count in counts.items():
        if term in vocab:
            vector[vocab[term]] = count
    vector *= np.asarray(calibration["idf"])
    norm = np.linalg.norm(vector)
    if not norm:
        return None
    similarity = np.asarray(calibration["centroids"]) @ (vector / norm)
    if not np.isfinite(similarity).all() or similarity.max() <= 0:
        return None
    return -1 if similarity.argmax() == 0 else 1


def question_proposition(turn, parsed, calibration):
    """Construct one addressed target proposition without consulting expected labels."""
    candidates = [
        f
        for f in frames(turn["text"], parsed, "partner")
        if f["question"] and f["target_relevant"]
    ]
    if len(candidates) != 1:
        return None, "no_unique_question"
    f = candidates[0]
    if f["sign"] < 0 or f["source_scoped"] or f["ambiguous_negation"]:
        return None, "ambiguous_scope"
    if f["channel"] not in {"intention", "past_action_report"}:
        return None, "unsupported_question_channel"
    conditions = [c["text"] for c in f["conditions"]]
    condition = frozen_condition(" ; ".join(conditions), calibration)
    times = re.findall(
        r"\b(?:yesterday|today|tomorrow|next week|last week)\b",
        turn["text"],
        re.IGNORECASE,
    )
    if condition is None:
        return None, "condition_unmapped"
    return {
        "query": "request_help",
        "actor": f["actor"],
        "mode": f["channel"],
        "condition": condition,
        "time_cues": [t.lower() for t in times],
        "question_id": turn["id"],
        "question_span": address(turn),
        "predicate_span": address(turn, f["start"], f["end"]),
        "condition_spans": [
            address(turn, c["start"], c["end"]) for c in f["conditions"]
        ],
    }, "question_open"


def answer_operation(text):
    """Recognize explicit reference/auxiliary patterns; no open ellipsis claim."""
    if RETRACT.fullmatch(text):
        return "retract", None, False
    kind, remainder = "answer", text
    for label, pattern in [("revise", REPAIR), ("alias", REPEAT)]:
        match = pattern.match(text)
        if match:
            kind, remainder = label, text[match.end() :]
            break
    match, bare = ANSWER.fullmatch(remainder), BARE.fullmatch(remainder)
    if match:
        auxiliary = match.group(1).lower()
        sign = -1 if "not" in auxiliary or "n't" in auxiliary else 1
        return kind, sign, True
    if bare:
        return kind, 1 if bare.group(1).lower() == "yes" else -1, False
    return "unresolved", None, False


def speech_observations(turn, parsed, recipient):
    """Record direct requests separately from the proposed act of helping."""
    found = []
    for f in frames(turn["text"], parsed, turn["speaker"]):
        if f["channel"] != "current_request_candidate" or f["source_scoped"]:
            continue
        found.append(
            {
                "id": f"{turn['id']}:{f['id']}",
                "speech_source": turn["speaker"],
                "speech_function": "direct_request_candidate",
                "recipient": recipient,
                "proposed_action_actor": recipient,
                "proposed_predicate": f["predicate"],
                "proposed_action_enacted": None,
                "span": address(turn, f["start"], f["end"]),
                "utterance_span": address(turn),
            }
        )
    return found


def numeric_snapshot(history):
    """Use the frozen M02 version ledger before taking typed finite means."""
    active = active_records(history)
    scores = {}
    for mode in ["past_action_report", "intention"]:
        for condition in [-1, 1]:
            records = [
                r
                for r in active["records"]
                if r["mode"] == mode and r["condition"] == condition
            ]
            scores[f"{mode}:{condition}"] = {
                "score": float(np.mean([r["value"] for r in records]))
                if records
                else None,
                "active_claims": len(records),
                "active_event_references": len({r["event_id"] for r in records}),
            }
    return {
        "active_ids": [r["record_id"] for r in active["records"]],
        "scores": scores,
        "observed_event_references": active["observed_events"],
        "stale_derived": active["stale_derived"],
    }


def read_dialogue(turns, parses, calibration):
    """Translate raw turn order into explicit, source-addressed version operations.

    event_id is a question-local event reference until an explicit reference
    licenses sameness. A new question alone does not establish that its physical
    event differs from earlier questions; that distinction remains in the output.
    """
    history, speech, snapshots, log = [], [], [], []
    question, last_claim = None, None
    participants = {t["speaker"] for t in turns}
    if not participants <= {"partner", "respondent"}:
        raise ValueError("This adapter requires explicit two-party turn roles")
    if len({t["id"] for t in turns}) != len(turns):
        raise ValueError("Turn IDs must be unique within a dialogue")
    for turn in turns:
        text, speaker = turn["text"], turn["speaker"]
        parsed = parses[text]
        if speaker == "partner":
            question, status = question_proposition(turn, parsed, calibration)
            log.append({"turn": turn["id"], "status": status})
        else:
            recipient = "partner" if "partner" in participants else None
            speech.extend(speech_observations(turn, parsed, recipient))
            kind, sign, explicit_self = answer_operation(text)
            before = {r["record_id"]: r for r in active_records(history)["records"]}
            parent = before.get(last_claim)
            referential = kind in {"revise", "alias", "retract"}
            if kind == "unresolved":
                status = "no_claim_update"
            elif referential and parent is None:
                status = "no_active_reference"
            elif kind in {"alias", "retract"}:
                if kind == "alias" and sign != parent["proposition_sign"]:
                    status = "repeat_content_conflict"
                else:
                    history.append(
                        {
                            "record_id": turn["id"],
                            "kind": kind,
                            "parent_id": parent["record_id"],
                            "span": address(turn),
                        }
                    )
                    status = kind
                    if kind == "retract":
                        last_claim = None
            else:
                proposition = deepcopy(
                    parent["proposition"] if kind == "revise" else question
                )
                if proposition is None:
                    status = "no_unique_question"
                else:
                    if explicit_self:
                        proposition["actor"] = "respondent"
                    if proposition["actor"] not in {"respondent", "other"}:
                        status = "unresolved_actor"
                    else:
                        event_id = (
                            parent["event_id"]
                            if kind == "revise"
                            else proposition["question_id"] + ":event_reference"
                        )
                        repeated = any(r.get("event_id") == event_id for r in history)
                        operation = (
                            "revise"
                            if kind == "revise"
                            else "repeat"
                            if repeated
                            else "new"
                        )
                        record = {
                            "record_id": turn["id"],
                            "kind": operation,
                            "event_id": event_id,
                            "block_id": "single_observed_session",
                            "condition": proposition["condition"],
                            "mode": proposition["mode"],
                            "value": float(
                                sign * (proposition["actor"] == "respondent")
                            ),
                            "proposition_sign": sign,
                            "proposition": proposition,
                            "span": address(turn),
                            "physical_event_identity_verified": False,
                        }
                        if operation == "revise":
                            record["parent_id"] = parent["record_id"]
                        history.append(record)
                        last_claim = turn["id"]
                        status = operation
            log.append({"turn": turn["id"], "status": status})
        snapshot = numeric_snapshot(history)
        snapshots.append(
            {
                "turn": turn["id"],
                **snapshot,
                "own_direct_requests": sum(
                    s["speech_source"] == "respondent" for s in speech
                ),
            }
        )
    return {
        "history": history,
        "speech": speech,
        "snapshots": snapshots,
        "operations": log,
        "measurement": "versioned reported expression plus observed authored speech candidate; not actual behavior",
    }
