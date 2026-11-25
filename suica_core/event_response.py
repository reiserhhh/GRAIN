#!/usr/bin/env python3
"""Opt-in addressed evidence for one proposition: adoption of a corrected premise.

The lexical reader intentionally recognises only explicit English forms. An
optional versioned semantic reader supplies judgments on the exact same prefix.
Liquid readouts are instrumentation, never votes about behavior.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from grain_research.dialogue_claim_ledger import address

from .liquid_instrument import digest


def _matches(pattern, text):
    return bool(re.fullmatch(pattern + r"[.!]?", text.strip(), re.IGNORECASE))


def _binding(dialogue, i):
    turn = dialogue["turns"][i]
    ref = address({**turn, "id": i})
    return dict(ref, source_id=turn.get("id"), timestamp=turn.get("timestamp"))


def _frame(dialogue, i, focal, old, new, trigger):
    return digest(
        [dialogue["id"], dialogue["turns"][: i + 1], trigger, focal, old, new]
    )


def inspect_event(doc, probe, liquid=None):
    ds = [d for d in doc["dialogues"] if d["id"] == probe["dialogue"]]
    if len(ds) != 1:
        raise ValueError("unknown or duplicate dialogue")
    d = ds[0]
    t, end = probe["trigger"], probe["cutoff"]
    focal, old, new = (probe[k] for k in ("focal", "old", "new"))
    if (
        type(t) is not int
        or type(end) is not int
        or not 0 <= t <= end < len(d["turns"])
        or not all(isinstance(s, str) and s.strip() for s in (old, new, focal))
        or old.casefold().strip() == new.casefold().strip()
    ):
        raise ValueError("invalid event window or competing premises")
    turns = d["turns"][: end + 1]
    observed = {x["speaker"] for x in turns}
    if len(observed | {focal}) > 2 or any(not s for s in observed):
        raise ValueError("a focal speaker and one partner are required")
    a, b = re.escape(old.strip()), re.escape(new.strip())
    trigger = _binding(d, t)
    english = doc["language"] == "en"
    explicit = english and _matches(
        rf"(?:it's {b}, not {a}|not {a}, but {b}|correction: {b}, not {a})",
        trigger["quote"],
    )
    trigger_id = _frame(d, t, focal, old, new, t)
    attestation = probe.get("correction")
    if attestation is not None and (
        attestation.get("prefix_id") != trigger_id
        or not attestation.get("version")
        or type(attestation.get("confirmed")) is not bool
    ):
        raise ValueError("unbound external correction annotation")
    out = {
        "dialogue": d["id"],
        "condition": d["condition"],
        "focal": focal,
        "speaker_correcting": trigger["speaker"],
        "old": old,
        "new": new,
        "cutoff": end,
        "trigger": trigger,
        "response": None,
        "lexical_reader": "explicit-correction-en-v1",
        "semantic_reader": None,
        "correction_reader": attestation,
        "trigger_prefix_id": trigger_id,
        "lexical_trigger": explicit,
        "liquid": None,
        "status": "insufficient",
        "opportunity": "trigger_unconfirmed",
        "reason": "no_explicit_partner_correction",
    }
    if attestation is not None and explicit and not attestation["confirmed"]:
        return dict(out, reason="conflicting_trigger_readers_review")
    if trigger["speaker"] == focal or not (
        explicit or (attestation and attestation["confirmed"])
    ):
        return out
    # Stop at an intervening partner turn: a new instruction may supersede the premise.
    if end == t or turns[t + 1]["speaker"] != focal:
        return dict(
            out, opportunity="reply_unobserved", reason="no_addressable_next_focal_turn"
        )
    i = t + 1
    reply = _binding(d, i)

    def adopts(value):
        return _matches(
            rf"(?:i(?:'ll| will) (?:use|choose|take|stick with)|i choose) {value}",
            reply["quote"],
        )

    lex = (
        "support"
        if english and adopts(b)
        else "refute"
        if english and adopts(a)
        else "unresolved"
    )
    out.update(
        response=reply,
        opportunity="observed",
        lexical_vote=lex,
        reason="reply_not_diagnostic",
        prefix_id=_frame(d, i, focal, old, new, t),
    )
    votes = {lex} - {"unresolved"}
    reader = probe.get("semantic")
    if reader is not None:
        if (
            not reader.get("version")
            or reader.get("prefix_id") != out["prefix_id"]
            or reader.get("vote") not in {"support", "refute", "unresolved"}
            or reader.get("actor") != focal
            or reader.get("kind", "candidate") not in {"candidate", "coded_response"}
        ):
            raise ValueError("unbound semantic annotation or invalid reader")
        out["semantic_reader"] = dict(reader)
        if reader.get("kind") == "coded_response":
            votes |= {reader["vote"]} - {"unresolved"}
    if len(votes) == 1:
        out.update(
            status=votes.pop(), reason="addressed_statement_not_independent_action"
        )
    elif votes:
        out["reason"] = "conflicting_readers_review"
    if liquid is not None:
        if liquid["kind"] != "dyad_reader_not_personal_factor_scores" or liquid[
            "input_digest"
        ] != digest(doc):
            raise ValueError("liquid trajectory is not bound to the input dialogue")
        trace = liquid["dialogues"][d["id"]]
        joint = trace["joint_trajectory"]
        slots = [j for j, (turn, _) in enumerate(joint["slots"]) if turn == i]
        if any(trace["speaker_at_slice"][j] != focal for j in slots):
            raise ValueError("speaker mismatch in liquid trajectory")
        out["liquid"] = {
            "reference": liquid["reference_digest"],
            "clock": liquid["clock"],
            "slots": [joint["slots"][j] for j in slots],
            "readout": [joint["liquid"][j] for j in slots],
        }
    return out


def joined_opportunity(events, feasible_votes):
    """One compound proposition (all clauses) about ONE action, not n opportunities.

    The caller supplies the jointly feasible binary truth rows, not independent
    marginal probabilities. Known atomic verdicts further filter these rows.
    """
    if not events or not all(e["opportunity"] == "observed" for e in events):
        raise ValueError("missing component opportunity")
    keys = {
        (
            e["dialogue"],
            e["focal"],
            e["condition"],
            e["response"]["turn_id"],
            e["response"]["sha256"],
            e["cutoff"],
        )
        for e in events
    }
    if len(keys) != 1:
        raise ValueError("compound event must share one action and observation cutoff")
    worlds = list(feasible_votes)
    if not worlds or any(
        len(w) != len(events) or any(type(b) is not int or b not in (0, 1) for b in w)
        for w in worlds
    ):
        raise ValueError("expected jointly feasible binary truth rows")
    remaining = [
        w
        for w in worlds
        if all(
            e["status"] == "insufficient" or bool(b) == (e["status"] == "support")
            for b, e in zip(w, events)
        )
    ]
    verdicts = {all(w) for w in remaining}
    status = (
        "support"
        if verdicts == {True}
        else "refute"
        if verdicts == {False}
        else "insufficient"
    )
    return dict(
        events[0],
        status=status,
        reason="joint_conjunction" if remaining else "joint_conflict_review",
        component_events=events,
        feasible_worlds=remaining,
    )


def conditional_card(events, focal, weights):
    """Rate bounds across observed opportunities and fixed target conditions."""
    if (
        not weights
        or any(not np.isfinite(w) or w < 0 for w in weights.values())
        or not np.isclose(sum(weights.values()), 1.0, rtol=0, atol=1e-9)
    ):
        raise ValueError("fixed reference weights must sum to one")
    seen, groups, missing = set(), defaultdict(Counter), Counter()
    for e in events:
        if e["focal"] != focal:
            continue
        if e["opportunity"] != "observed":
            missing[e["opportunity"]] += 1
            continue
        key = (e["dialogue"], e["response"]["turn_id"])
        if key in seen:
            raise ValueError(
                "shared action: do not count overlapping probes as independent"
            )
        seen.add(key)
        groups[e["condition"]][e["status"]] += 1
    bounds = [0.0, 0.0]
    rows = {}
    for k in sorted(groups.keys() | weights.keys()):
        c = groups[k]
        n = sum(c.values())
        lo, hi = (c["support"] / n, 1.0 - c["refute"] / n) if n else (0.0, 1.0)
        rows[k] = {
            "counts": dict(c),
            "n": n,
            "rate_bounds": [lo, hi],
            "observed_opportunity_share": n / len(seen) if seen else None,
        }
        bounds[0] += weights.get(k, 0.0) * lo
        bounds[1] += weights.get(k, 0.0) * hi
    return {
        "focal": focal,
        "conditions": rows,
        "weights": weights,
        "reference_rate_bounds": bounds,
        "unconfirmed_or_no_observed_reply": dict(missing),
    }


def later_check(event, action):
    """Display an external action record separately; never change the as-of vote."""
    if (
        action.get("dialogue") != event["dialogue"]
        or type(action.get("after_turn")) is not int
        or action["after_turn"] <= event["cutoff"]
        or not all(
            isinstance(action.get(k), str) and action[k]
            for k in ("source", "id", "actor")
        )
    ):
        raise ValueError("expected an identified later action in this dialogue")
    side = "unresolved"
    if action["actor"] == event["focal"]:
        side = {event["old"]: "old", event["new"]: "new"}.get(
            action.get("choice"), "unresolved"
        )
    return {
        "event_as_of_status": event["status"],
        "cutoff": event["cutoff"],
        "later_action": action,
        "focal_choice_of_premise": side,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--liquid", type=Path)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    liquid = json.loads(args.liquid.read_text()) if args.liquid else None
    events = [inspect_event(data["doc"], probe, liquid) for probe in data["probes"]]
    cards = [
        conditional_card(events, f, data["weights"])
        for f in sorted({e["focal"] for e in events})
    ]
    checks = [
        later_check(e, p["later_action"])
        for e, p in zip(events, data["probes"])
        if "later_action" in p
    ]
    print(
        json.dumps(
            {"events": events, "cards": cards, "later_checks": checks},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
