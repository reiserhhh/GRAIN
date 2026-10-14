#!/usr/bin/env python3
"""Passive event/mention composition on explicit, versioned behavioral records."""

from __future__ import annotations

import itertools
import math

import numpy as np

from grain_research.evidence_history import active_records

REGIMES = (
    "balanced",
    "alias_focus",
    "distinct_focus",
    "other_focus",
    "revision",
    "revision_alias",
    "retracted",
    "other_only",
)
CONTRASTS = (
    ("alias_minus_balanced", "alias_focus", "balanced"),
    ("distinct_minus_alias", "distinct_focus", "alias_focus"),
    ("distinct_minus_balanced", "distinct_focus", "balanced"),
    ("other_minus_balanced", "other_focus", "balanced"),
    ("revision_alias_minus_revision", "revision_alias", "revision"),
    ("revision_minus_balanced", "revision", "balanced"),
    ("retracted_minus_balanced", "retracted", "balanced"),
    ("other_only_minus_balanced", "other_only", "balanced"),
)


def record(rid, event, family, polarity, actor="A", kind="new", parent=None):
    """Create explicit design metadata, never infer an event from its text."""
    result = {
        "record_id": rid,
        "event_id": event,
        "family": family,
        "polarity": polarity,
        "actor": actor,
        "kind": kind,
        "condition": 1 if family == 0 else -1,
        "block_id": f"family_{family}",
    }
    if parent is not None:
        result["parent_id"] = parent
    return result


def action_line(event, actor, action):
    """Use the same syntax for repeated and distinct occasion descriptions."""
    return f"On occasion {event}, Person {actor} {action}."


def account(spec, target, focal_family, polarity, order, regime):
    """Construct balanced/repeated/new/other/corrected histories without labels."""
    if regime not in REGIMES or order not in (0, 1) or polarity not in (0, 1):
        raise ValueError("Declared design values required")
    other = 1 - focal_family
    words = spec["behaviours"][target]
    actor = "B" if regime == "other_only" else "A"
    focus = [record("r1", "K1", focal_family, polarity, actor)]
    companion = [record("r2", "K2", other, 1 - polarity, actor)]
    if regime in ("alias_focus", "revision_alias", "distinct_focus"):
        for rid, event in (("r3", "K3"), ("r4", "K4")):
            is_alias = regime != "distinct_focus"
            focus.append(
                record(
                    rid,
                    "K1" if is_alias else event,
                    focal_family,
                    polarity,
                    actor,
                    "alias" if is_alias else "new",
                    "r1" if is_alias else None,
                )
            )
    if regime == "other_focus":
        focus.extend(
            record(r, e, focal_family, polarity, "B")
            for r, e in (("r3", "K3"), ("r4", "K4"))
        )
    blocks = (focus, companion) if order == 0 else (companion, focus)
    history = [r for b in blocks for r in b]
    lines = [
        action_line(r["event_id"], r["actor"], words[r["family"]][r["polarity"]])
        for r in history
    ]
    if regime in ("revision", "revision_alias"):
        history.append(
            record("r5", "K1", focal_family, 1 - polarity, kind="revise", parent="r1")
        )
        lines.append(
            "Correction about occasion K1: Person A instead "
            + words[focal_family][1 - polarity]
            + ". The earlier K1 description was mistaken and is replaced by this one."
        )
    if regime == "retracted":
        history.append(
            record("r5", "K1", focal_family, polarity, kind="retract", parent="r1")
        )
        lines.append(
            "The earlier description of occasion K1 is withdrawn; "
            "no claim about that occasion is available."
        )
    active_records(history)
    return {
        "role": "account",
        "target": target,
        "family": focal_family,
        "polarity": polarity,
        "order": order,
        "regime": regime,
        "body": "\n".join(lines),
        "history": history,
    }


def documents(spec):
    """Return all 24 single events before all 384 joint accounts."""
    if spec["regimes"] != list(REGIMES):
        raise ValueError("Eight unchanged regimes required")
    docs = []
    for target, family, polarity in itertools.product(spec["targets"], (0, 1), (0, 1)):
        r = record("r1", "K1", family, polarity)
        docs.append(
            {
                "role": "atom",
                "target": target,
                "family": family,
                "polarity": polarity,
                "order": -1,
                "regime": "single_event",
                "body": action_line(
                    "K1", "A", spec["behaviours"][target][family][polarity]
                ),
                "history": [r],
            }
        )
    for values in itertools.product(spec["targets"], (0, 1), (0, 1), (0, 1), REGIMES):
        docs.append(account(spec, *values))
    return [{**r, "unit": i} for i, r in enumerate(docs)]


def channels(history, atom_values):
    """Keep qualified events and descriptive mentions distinct; absent is null."""
    ledger = active_records(history)
    events = [r for r in ledger["records"] if r["actor"] == "A"]
    mentions = [
        r
        for r in history
        if r["actor"] == "A" and r["kind"] in {"new", "alias", "revise"}
    ]
    if len({r["event_id"] for r in events}) != len(events):
        raise ValueError("This finite design requires one current claim per event")

    def mean(records):
        if not records:
            return None
        values = [float(atom_values[r["family"], r["polarity"]]) for r in records]
        if not np.isfinite(values).all():
            raise ValueError("Finite isolated-event observations required")
        return math.fsum(values) / len(values)

    return {
        "event_profile": mean(events),
        "mention_profile": mean(mentions),
        "own_events": len(events),
        "own_mentions": len(mentions),
        "total_reports": len(history),
        "stale_derived": ledger["stale_derived"],
        "active_event_ids": sorted(r["event_id"] for r in events),
    }


def displacement(a, b, c):
    """Describe alias and new-event endpoint displacements, not a causal circuit."""
    d_alias, d_new = np.asarray(b, float) - a, np.asarray(c, float) - b
    n_alias, n_new = float(np.linalg.norm(d_alias)), float(np.linalg.norm(d_new))
    return {
        "alias_norm": n_alias,
        "distinct_minus_alias_norm": n_new,
        "cosine": float(d_alias @ d_new / (n_alias * n_new))
        if n_alias > 0 and n_new > 0
        else None,
    }


def summarize(rows):
    """Report paired descriptive effects without treating templates as people."""
    result = {"paired_cases": len(rows)}
    for key in ("score", "log_odds", "event_profile", "mention_profile"):
        x = np.array([r[key] for r in rows if r[key] is not None], float)
        result[key + "_mean"] = float(x.mean()) if len(x) else None
        result[key + "_rms"] = float(np.sqrt(np.mean(x * x))) if len(x) else None
        result[key + "_observed_cases"] = len(x)
    oriented = [r["oriented_score"] for r in rows if r["oriented_score"] is not None]
    result["oriented_positive"] = sum(x > 0 for x in oriented)
    result["oriented_cases"] = len(oriented)
    return result
