#!/usr/bin/env python3
"""Evidence-qualified event aggregation for frozen paired-response readouts.

Event identity and source qualification are declared by an evidence ledger, not
inferred from vector similarity. The numeric scorer never accepts author labels.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from grain_research.evidence_history import active_records

ACTION = re.compile(
    r"\bi\s+(organized|organised|coordinated|arranged|scheduled|invited|gathered|contacted|suggested)\b",
    re.IGNORECASE,
)
TEAM = re.compile(
    r"\b(we|our|us|team|group|meeting|together|friends|people|everyone|plan|project)\b",
    re.IGNORECASE,
)
BASE_VERBS = {
    "organized": "organize",
    "organised": "organise",
    "coordinated": "coordinate",
    "arranged": "arrange",
    "scheduled": "schedule",
    "invited": "invite",
    "gathered": "gather",
    "contacted": "contact",
    "suggested": "suggest",
}


def candidate_slice(text: str) -> dict:
    """Locate an explicit past-action candidate, without declaring it true.

    This narrow lexical inventory is frozen before scoring. It is candidate
    recall, not a complete initiation/coordination or ownership classifier.
    Offsets preserve the exact original words; unsupported forms remain absent.
    """
    hit = ACTION.search(text)
    if hit is None or not TEAM.search(text):
        return {"status": "no_candidate"}
    end_hit = re.search(r"[.!?\n]", text[hit.end() :])
    end = hit.end() + end_hit.start() if end_hit else len(text)
    fragment = text[hit.start() : end].strip().rstrip(" ,;:")
    if len(fragment.split()) < 4 or len(fragment.split()) > 56:
        return {"status": "unsupported_span_length"}
    prefix = text[: hit.start()]
    quote_prefix = prefix[prefix.rfind("\n") + 1 :]
    if quote_prefix.lstrip().startswith(">") or quote_prefix.count('"') % 2:
        return {"status": "quotation_candidate"}
    return {
        "status": "past_self_action_candidate",
        "start": hit.start(),
        "end": end,
        "fragment": fragment,
        "verb": hit.group(1).lower(),
        "qualification": "candidate_not_human_verified",
        "condition": None,
    }


def probe_variants(fragment: str, verb: str) -> dict[str, str]:
    """Declare counterfactual probes anchored in one natural action fragment.

    Negation means denial of this action, not a claim that the author waited.
    Reframed probes are synthetic edits, never additional observed events.
    """
    match = ACTION.match(fragment)
    if match is None or match.group(1).lower() != verb:
        raise ValueError("Probe requires an aligned explicit action span")
    complement = fragment[match.end() :].strip()
    base = BASE_VERBS[verb]
    own = fragment + "."
    other = f"A collaborator said: '{own}' My own response is not reported."
    return {
        "fragment": own,
        "own_envelope": f"This is my own account: {own} I am describing my own response, not another person's.",
        "denied_action": f"I did not {base} {complement}.",
        "other_account": other,
        "hypothetical": f"If this happened again, I would {base} {complement}.",
        "wait_only": "I waited for someone else to take the first step in coordinating the work.",
        "mixed_other": f"I waited for someone else to take the first step in coordinating the work. {other}",
    }


def replay_qualified(history: list[dict]) -> dict:
    """Route an author ledger, then replay verified versioned dependencies.

    Unknown context cannot enter an endpoint-specific score. Keep such records
    in a separate ledger, rather than converting them to zero or assigning a
    convenient condition. Ineligible evidence and aliases remain in provenance.
    """
    authors = {r["author_id"] for r in history}
    if len(authors) > 1:
        raise ValueError("One author per evidence ledger")
    active, seen, annotations, event_context = {}, {}, [], {}
    observed_qualified_events = set()
    for original in history:
        row = dict(original)
        rid = row["record_id"]
        if rid in seen:
            raise ValueError("Duplicate record ID")
        kind = row["kind"]
        if kind in {"alias", "retract", "summary"}:
            parents = row.get("parents", [row.get("parent_id")])
            if not parents or not set(parents) <= set(active):
                raise ValueError("Reference must name active claims")
            if kind == "retract":
                del active[row["parent_id"]]
            else:
                annotations.append(row)
        else:
            if kind not in {"new", "repeat", "revise"}:
                raise ValueError("Unknown evidence operation")
            event = row["event_id"]
            if kind == "new" and event in event_context:
                raise ValueError("Existing event is not an independent new event")
            if kind == "repeat" and event not in event_context:
                raise ValueError("Repeat has no observed event")
            if kind == "revise":
                parent = active.get(row["parent_id"])
                if parent is None or parent["event_id"] != event:
                    raise ValueError("Revision requires active same-event parent")
                del active[row["parent_id"]]
            context = (row.get("condition"), row["block_id"])
            if event in event_context:
                old_c, old_block = event_context[event]
                if old_block != context[1] or (
                    old_c is not None and context[0] is not None and old_c != context[0]
                ):
                    raise ValueError("Same event has conflicting context")
                if context[0] is None:
                    context = (old_c, context[1])
            event_context[event] = context
            if row.get("qualification") == "own_actual" and row.get("condition") in (
                -1,
                1,
            ):
                value = row.get("value")
                if value is None or not np.isfinite(value):
                    raise ValueError("Qualified records require finite scalar scores")
                observed_qualified_events.add(event)
            active[rid] = row
        seen[rid] = row
    routed, excluded, routed_events = [], [], set()
    for row in active.values():
        if row.get("qualification") != "own_actual":
            excluded.append({**row, "reason": "not_own_actual"})
        elif row.get("condition") not in (-1, 1):
            excluded.append({**row, "reason": "unknown_condition"})
        else:
            event = row["event_id"]
            # Replay versions before routing: a newly ineligible correction must
            # remove its old eligible parent rather than silently keep its score.
            routed.append(
                {**row, "kind": "repeat" if event in routed_events else "new"}
            )
            routed_events.add(event)
    state = (
        active_records(routed)
        if routed
        else {"records": [], "history": [], "active_events": 0, "observed_events": 0}
    )
    stale = [
        r["record_id"]
        for r in annotations
        if not set(r.get("parents", [r.get("parent_id")])) <= set(active)
    ]
    return {
        **state,
        "history": list(seen.values()),
        "stale_derived": stale,
        "observed_events": len(observed_qualified_events),
        "declared_events": len(event_context),
        "excluded": excluded,
        "input_records": len(history),
    }


def fixed_reference_profile(records: list[dict], mass_positive=0.5) -> dict:
    """Estimate event-balanced endpoints, retaining choice and missing coverage.

    Several reports of one event get one combined event value. This direct mean
    is an observation profile, not a posterior or a psychological trait score.
    """
    if not 0 <= mass_positive <= 1:
        raise ValueError("Reference mass must lie in [0, 1]")
    events = {}
    for row in records:
        c = row["condition"]
        if c not in (-1, 1):
            raise ValueError("Declared endpoint condition required")
        key = row["event_id"]
        if key in events and events[key]["condition"] != c:
            raise ValueError("Same event has conflicting conditions")
        events.setdefault(key, {"condition": c, "values": []})["values"].append(
            float(row["value"])
        )
    endpoint = {}
    counts = Counter(event["condition"] for event in events.values())
    for c in (-1, 1):
        values = [
            np.mean(event["values"])
            for event in events.values()
            if event["condition"] == c
        ]
        endpoint[c] = float(np.mean(values)) if values else None
    both = all(endpoint[c] is not None for c in (-1, 1))
    mass = mass_positive
    return {
        "endpoints": {str(k): v for k, v in endpoint.items()},
        "level": (1 - mass) * endpoint[-1] + mass * endpoint[1] if both else None,
        "response": (endpoint[1] - endpoint[-1]) / 2 if both else None,
        "condition_choice_positive": counts[1] / len(events) if events else None,
        "events": len(events),
        "event_counts": {str(c): counts[c] for c in (-1, 1)},
        "unobserved_conditions": [c for c in (-1, 1) if endpoint[c] is None],
        "reference_mass_positive": mass,
    }


def panel_schedule() -> list[dict]:
    """Four blocks, both conditions, two distinct events, two reports per event."""
    return [
        {
            "record_id": f"b{b}:c{c}:e{e}:r{r}",
            "event_id": f"b{b}:c{c}:e{e}",
            "block_id": f"b{b}",
            "condition": c,
            "kind": "new" if r == 0 else "repeat",
            "qualification": "own_actual",
            "author_id": "schedule",
        }
        for b in range(4)
        for c in (-1, 1)
        for e in range(2)
        for r in range(2)
    ]


def simulate_panels(lookup, families, n, seed):
    """Generate distinct-event reports using frozen observed text-score values.

    Author parameters generate event probabilities but are returned separately
    for evaluation. Family content is shared by reports, not by distinct events.
    The observation generator adds declared block and report perturbations to
    make covariance components estimable; it is not a model of human authors.
    """
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.3, 0.3, n)
    schedule = panel_schedule()
    values = np.empty((n, len(schedule)))
    for u in range(n):
        blocks = rng.normal(0, 0.12, 4)
        event = {}
        for j, row in enumerate(schedule):
            key = row["event_id"]
            c = row["condition"]
            if key not in event:
                p = 1 if rng.random() < (1 + a[u] + b[u] * c) / 2 else -1
                event[key] = (p, int(rng.choice(families)))
            p, f = event[key]
            register = -1 if row["kind"] == "new" else 1
            values[u, j] = (
                lookup[f, p, register, c]
                + blocks[int(row["block_id"][1:])]
                + rng.normal(0, 0.04)
            )
    return {"values": values, "records": schedule}, {"level_response": np.c_[a, b]}
