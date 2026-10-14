#!/usr/bin/env python3
"""Target-generic dialogue ledger for coverage diagnosis on natural antecedent pairs.

Reuses the frozen explicit-reference rules, answer auxiliaries, actor resolution
and the M02 version account. It differs from ``dialogue_claim_ledger`` in two
declared ways: (1) a question does not need the authored help-request target;
the score key is (channel, predicate lemma) read from the unique question frame;
(2) a condition clause is kept as an addressed span (normalized text key) rather
than mapped onto the two frozen source prototypes. A reply whose polar prefix is
followed by further content is recorded as a candidate, never scored. This is a
diagnostic instrument, not a new measurement claim.
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy

import numpy as np

from grain_research.dialogue_claim_ledger import address, answer_operation
from grain_research.proposition_response import frames

POLAR_PREFIX = re.compile(r"^\s*(yes|no|yeah|yep|nope|nah)\b", re.IGNORECASE)
BINDABLE = {
    "intention",
    "past_action_report",
    "preference",
    "ability",
    "ability_candidate",
    "current_experience_candidate",
    "prospective",
    "unspecified_expression",
}


def normalize_condition(text):
    """Address-preserving condition key: lowercase, collapsed spaces; no mapping."""
    return re.sub(r"\s+", " ", text.strip().lower()) if text else None


def generic_question(turn, parsed):
    """Return one bindable question proposition or a typed reason for none."""
    found = frames(turn["text"], parsed, "partner")
    questions = [f for f in found if f["question"]]
    eligible = [
        f
        for f in questions
        if f["actor"] in {"respondent", "other"}
        and not f["source_scoped"]
        and not f["ambiguous_negation"]
        and f["sign"] > 0
        and f["channel"] in BINDABLE
    ]
    account = {
        "frames": len(found),
        "question_frames": len(questions),
        "eligible_frames": len(eligible),
        "channels": dict(Counter(f["channel"] for f in questions)),
        "actors": dict(Counter(f["actor"] for f in questions)),
        "condition_clause_frames": sum(bool(f["conditions"]) for f in questions),
        "target_relevant_frames": sum(f["target_relevant"] for f in questions),
    }
    if not questions:
        return None, "no_question_frame", account
    if len(eligible) == 0:
        return None, "question_not_bindable", account
    if len(eligible) > 1:
        return None, "multiple_bindable_questions", account
    f = eligible[0]
    clause = " ; ".join(c["text"] for c in f["conditions"]) if f["conditions"] else ""
    proposition = {
        "query": f"{f['channel']}:{f['predicate']}",
        "actor": f["actor"],
        "mode": f["channel"],
        "predicate": f["predicate"],
        "condition": normalize_condition(clause),
        "condition_addressed": bool(f["conditions"]),
        "question_id": turn["id"],
        "question_span": address(turn),
        "predicate_span": address(turn, f["start"], f["end"]),
        "condition_spans": [
            address(turn, c["start"], c["end"]) for c in f["conditions"]
        ],
    }
    return proposition, "question_open", account


def classify_reply(text):
    """Frozen answer operation plus a non-scored polar-prefix candidate class."""
    kind, sign, explicit_self = answer_operation(text)
    match = POLAR_PREFIX.match(text or "")
    if kind == "unresolved" and match:
        polarity = 1 if match.group(1).lower() in {"yes", "yeah", "yep"} else -1
        return "polar_prefix_with_content", polarity, False
    return kind, sign, explicit_self


def replay_versions(history):
    """Diagnostic replay of the M02 version operations without the endpoint-condition gate.

    Mirrors ``evidence_history.active_records`` (new/repeat/revise/alias/retract) but
    accepts addressed or absent conditions, because the generic ledger keeps
    conditions as spans rather than declared -1/+1 endpoints. Tested against the
    frozen function on conditioned histories.
    """
    seen, active, events = set(), {}, {}
    for original in history:
        r = deepcopy(original)
        identity, kind = r["record_id"], r["kind"]
        if identity in seen:
            raise ValueError("Duplicate record ID; use an explicit alias")
        if kind in {"new", "repeat", "revise"}:
            key = r["event_id"]
            if kind == "new" and key in events:
                raise ValueError("Existing event is not an independent new event")
            if kind == "repeat" and key not in events:
                raise ValueError("Repeat has no observed event")
            if kind == "revise":
                old = active.get(r["parent_id"])
                if old is None or old["event_id"] != key:
                    raise ValueError("Revision requires an active same-event parent")
                del active[r["parent_id"]]
            events[key] = r.get("condition")
            active[identity] = r
        elif kind in {"alias", "retract"}:
            if r["parent_id"] not in active:
                raise ValueError("Reference must name an active claim")
            if kind == "retract":
                del active[r["parent_id"]]
        else:
            raise ValueError("Unknown evidence operation")
        seen.add(identity)
    return {"records": list(active.values()), "observed_events": sorted(events)}


def snapshot(history):
    """Finite means per (mode:predicate:condition) over the active version account."""
    active = replay_versions(history)
    scores = {}
    for record in active["records"]:
        if record.get("proposition", {}).get("actor", "respondent") != "respondent":
            continue
        key = f"{record['mode']}:{record['predicate']}:{record['condition']}"
        scores.setdefault(key, []).append(record["value"])
    return {
        "active_ids": [r["record_id"] for r in active["records"]],
        "scores": {
            k: {"score": float(np.mean(v)), "active_claims": len(v)}
            for k, v in scores.items()
        },
        "observed_event_references": active["observed_events"],
    }


def elaboration_conflict(text, parsed):
    """Does the content after the polar particle carry a locally negated frame?"""
    found = frames(text, parsed, "respondent")
    return any(f["local_negation"] for f in found), len(found)


def polarity_scope_decision(text, parsed, proposition, sign):
    """Keep unrelated negation; abstain on contradiction, repair or unresolved scope.

    Predicate identity is only a local syntactic match, not event coreference.
    This rule therefore abstains rather than overwriting a particle with an
    inferred factual answer. No change is made to frozen historical artifacts.
    """
    if re.search(
        r"\b(actually|correction|i mean|take that back)\b", text, re.IGNORECASE
    ):
        return "repair_requires_scope"
    found = frames(text, parsed, "respondent")
    own = [f for f in found if f["actor"] == "respondent" and not f["source_scoped"]]
    same = [f for f in own if f["predicate"] == proposition["predicate"]]
    if any(f["ambiguous_negation"] or f["sign"] != sign for f in same):
        return "same_predicate_conflict"
    if (
        re.search(r"\b(not|never|nothing|neither)\b|n't\b", text, re.IGNORECASE)
        and not own
    ):
        return "unresolved_negation"
    return "retain_particle"


def read_dialogue_generic(turns, parses, *, polar_binding=False, scope_guard=False):
    """Two-party turns -> generic proposition ledger with typed failure reasons.

    polar_binding=True additionally binds a reply that starts with a polar
    particle (Roelofsen & Farkas 2015: the particle answers the antecedent
    proposition) to the unique open question, flags the elaboration, and records
    whether the elaboration contains a negated frame. Such records carry
    confidence 'polar_with_elaboration' and are counted separately.
    """
    history, snapshots, log, accounts = [], [], [], []
    question, last_claim = None, None
    participants = {t["speaker"] for t in turns}
    if not participants <= {"partner", "respondent"}:
        raise ValueError("This adapter requires explicit two-party turn roles")
    for turn in turns:
        parsed = parses[turn["text"]]
        if turn["speaker"] == "partner":
            question, status, account = generic_question(turn, parsed)
            accounts.append({"turn": turn["id"], **account})
            log.append({"turn": turn["id"], "status": status})
        else:
            kind, sign, explicit_self = classify_reply(turn["text"])
            before = {r["record_id"]: r for r in replay_versions(history)["records"]}
            parent = before.get(last_claim)
            referential = kind in {"revise", "alias", "retract"}
            if kind == "polar_prefix_with_content" and not (
                polar_binding and question is not None
            ):
                status = "polar_prefix_with_content"
            elif (
                kind == "polar_prefix_with_content"
                and polar_binding
                and question is not None
            ):
                conflict, extra = elaboration_conflict(turn["text"], parsed)
                decision = (
                    polarity_scope_decision(turn["text"], parsed, question, sign)
                    if scope_guard
                    else "retain_particle"
                )
                if decision != "retain_particle":
                    log.append(
                        {
                            "turn": turn["id"],
                            "status": decision,
                            "answer_kind": kind,
                            "polarity": sign,
                        }
                    )
                    snapshots.append({"turn": turn["id"], **snapshot(history)})
                    continue
                proposition = deepcopy(question)
                event_id = proposition["question_id"] + ":event_reference"
                operation = (
                    "repeat"
                    if any(r.get("event_id") == event_id for r in history)
                    else "new"
                )
                history.append(
                    {
                        "record_id": turn["id"],
                        "kind": operation,
                        "event_id": event_id,
                        "block_id": "single_observed_thread",
                        "condition": proposition["condition"],
                        "mode": proposition["mode"],
                        "predicate": proposition["predicate"],
                        "value": float(sign * (proposition["actor"] == "respondent")),
                        "proposition_sign": sign,
                        "proposition": proposition,
                        "span": address(turn),
                        "confidence": "polar_with_elaboration",
                        "elaboration_negated_frame": conflict,
                        "elaboration_frames": extra,
                        "physical_event_identity_verified": False,
                    }
                )
                last_claim = turn["id"]
                status = (
                    "new_polar_with_elaboration"
                    if not conflict
                    else "new_polar_with_elaboration_negation_flag"
                )
            elif kind == "unresolved":
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
                    event_id = (
                        parent["event_id"]
                        if kind == "revise"
                        else proposition["question_id"] + ":event_reference"
                    )
                    operation = (
                        "revise"
                        if kind == "revise"
                        else (
                            "repeat"
                            if any(r.get("event_id") == event_id for r in history)
                            else "new"
                        )
                    )
                    record = {
                        "record_id": turn["id"],
                        "kind": operation,
                        "event_id": event_id,
                        "block_id": "single_observed_thread",
                        "condition": proposition["condition"],
                        "mode": proposition["mode"],
                        "predicate": proposition["predicate"],
                        "value": float(sign * (proposition["actor"] == "respondent")),
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
            log.append(
                {
                    "turn": turn["id"],
                    "status": status,
                    "answer_kind": kind,
                    "polarity": sign,
                }
            )
        snapshots.append({"turn": turn["id"], **snapshot(history)})
    return {
        "history": history,
        "snapshots": snapshots,
        "operations": log,
        "question_accounts": accounts,
        "measurement": "generic (channel, predicate, addressed condition) finite means; diagnostic only",
    }
