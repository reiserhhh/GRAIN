#!/usr/bin/env python3
"""Post-r1 target scoping without deleting multifunctional dialogue events.

This finite adapter distinguishes a request's proposed action from its general
directive form. It is not a full speech-act classifier or a replacement for the
original r1 generic-request count.
"""

from __future__ import annotations

from copy import deepcopy

from grain_research.dialogue_claim_ledger import REPAIR


def scope_speech(events):
    """Retain every candidate and attach nonexclusive, addressed target fields."""
    output = []
    for original in events:
        event = deepcopy(original)
        text = event["utterance_span"]["quote"]
        predicate = event["proposed_predicate"]
        repair = bool(REPAIR.match(text))
        event["explicit_previous_answer_repair"] = repair
        event["task_assistance_candidate"] = predicate in {"help", "assist"}
        event["discourse_permission_candidate"] = predicate == "let" and repair
        event["scope_basis"] = (
            "declared predicate target and explicit reference, not inferred personality"
        )
        output.append(event)
    return output


def scoped_prefixes(turns, events):
    """Keep broad, target-specific and discourse counts in separate channels."""
    events = scope_speech(events)
    seen, rows = set(), []
    for turn in turns:
        seen.add(turn["id"])
        observed = [e for e in events if e["span"]["turn_id"] in seen]
        rows.append(
            {
                "turn": turn["id"],
                "all_directive_candidates": len(observed),
                "task_assistance_candidates": sum(
                    e["task_assistance_candidate"] for e in observed
                ),
                "discourse_permission_candidates": sum(
                    e["discourse_permission_candidate"] for e in observed
                ),
            }
        )
    return rows
