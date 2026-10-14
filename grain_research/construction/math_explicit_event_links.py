#!/usr/bin/env python3
"""Actor-scoped event partitions with constant action/mention content."""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter

import numpy as np

SCHEMES = ("focus_alias", "companion_alias")
ORDERS = (("K1", "K2", "K3", "K4"), ("K4", "K3", "K2", "K1"))
WORD = re.compile(r"\w+|[^\w\s]")


def word_bag(text):
    """Count raw case-sensitive words and punctuation, without preprocessing."""
    return Counter(WORD.findall(text))


def declaration(actor, syntax, scheme):
    """Return the fixed scope and two word-matched relation statements."""
    if actor not in ("A", "B") or syntax not in (0, 1) or scheme not in SCHEMES:
        raise ValueError("Declared actor, syntax and scheme required")
    focus_same = scheme == "focus_alias"
    if syntax == 0:
        same = "Occasion {a} is the same event as occasion {b}."
        different = "Occasion {a} is a different event from occasion {b}."
    else:
        same = "{a} and {b} refer to one event."
        different = "{a} and {b} refer to separate events."
    first = (same if focus_same else different).format(a="K3", b="K1")
    second = (different if focus_same else same).format(a="K4", b="K2")
    scope = f"The following event identity statements apply only to Person {actor}'s occasions."
    return f"{scope}\n{first}\n{second}"


def canonical_events(records, actor, scheme):
    """Build an equivalence graph; presentation order never defines parenthood."""
    if actor not in ("A", "B") or scheme not in SCHEMES:
        raise ValueError("Declared relation scope required")
    if len(records) != 4 or {r["occasion_code"] for r in records} != set(ORDERS[0]):
        raise ValueError("Four unique surface occasions required")
    if len({r["record_id"] for r in records}) != 4:
        raise ValueError("Unique record identities required")
    by_code = {r["occasion_code"]: r for r in records}
    parent_code, alias_code = ("K1", "K3") if scheme == "focus_alias" else ("K2", "K4")
    result = []
    for r in records:
        code = r["occasion_code"]
        merged = r["actor"] == actor and code == alias_code
        root = parent_code if merged else code
        item = {**r, "event_id": r["actor"] + ":" + root}
        item["kind"] = "alias" if merged else "new"
        if merged:
            parent = by_code[parent_code]
            if any(parent[k] != r[k] for k in ("actor", "family", "polarity")):
                raise ValueError("Conflicting content cannot be silently deduplicated")
            item["parent_id"] = parent["record_id"]
        result.append(item)
    validate_history(result)
    return result


def validate_history(history):
    """Validate static alias edges, including references to later narrative rows."""
    by_id = {r["record_id"]: r for r in history}
    if len(by_id) != len(history) or not history:
        raise ValueError("Nonempty unique record identities required")
    for r in history:
        if r["event_id"].split(":", 1)[0] != r["actor"]:
            raise ValueError("Event identity must be actor scoped")
        if r["kind"] == "alias":
            parent = by_id.get(r.get("parent_id"))
            if parent is None or parent["kind"] != "new":
                raise ValueError("Explicit original parent required")
            if any(
                parent[k] != r[k] for k in ("actor", "event_id", "family", "polarity")
            ):
                raise ValueError("Alias identity/content conflict")
        elif r["kind"] != "new" or "parent_id" in r:
            raise ValueError("This static design has only new records and aliases")
    for event in {r["event_id"] for r in history}:
        if sum(r["event_id"] == event and r["kind"] == "new" for r in history) != 1:
            raise ValueError("Exactly one original per declared event required")


def document(behaviours, content, family, polarity, order, syntax, actor, scheme):
    """Keep the four action lines fixed and swap only declared event equivalence."""
    if family not in (0, 1) or polarity not in (0, 1) or order not in (0, 1):
        raise ValueError("Declared family, polarity and order required")
    records, lines = [], []
    for code in ORDERS[order]:
        focal = code in ("K1", "K3")
        f, p = (family, polarity) if focal else (1 - family, 1 - polarity)
        records.append(
            {
                "record_id": "r" + code[1],
                "occasion_code": code,
                "actor": "A",
                "family": f,
                "polarity": p,
                "condition": 1 if f == 0 else -1,
            }
        )
        lines.append(f"On occasion {code}, Person A {behaviours[content][f][p]}.")
    action = "\n".join(lines)
    relation = declaration(actor, syntax, scheme)
    return {
        "content": content,
        "query": content,
        "family": family,
        "polarity": polarity,
        "order": order,
        "syntax": syntax,
        "declaration_actor": actor,
        "scheme": scheme,
        "action_block": action,
        "declaration": relation,
        "body": action + "\n\n" + relation,
        "history": canonical_events(records, actor, scheme),
        "declared_relations": [
            {"actor": actor, "a": "K3", "b": "K1", "same": scheme == "focus_alias"},
            {"actor": actor, "a": "K4", "b": "K2", "same": scheme == "companion_alias"},
        ],
    }


def documents(behaviours, targets):
    """Enumerate the complete 384-arm roster before any new observation."""
    rows = []
    for values in itertools.product(
        targets, (0, 1), (0, 1), (0, 1), (0, 1), ("A", "B"), SCHEMES
    ):
        rows.append({"unit": len(rows), **document(behaviours, *values)})
    return rows


def paired_roster(rows):
    """Pair both schemes for every fixed actor, syntax, order and action block."""
    keys = ("content", "family", "polarity", "order", "syntax", "declaration_actor")
    lookup = {tuple(r[k] for k in keys) + (r["scheme"],): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate material condition")
    pairs = []
    for key in sorted({k[:-1] for k in lookup}):
        a, b = (lookup[key + (scheme,)] for scheme in SCHEMES)
        if a["action_block"] != b["action_block"] or word_bag(a["body"]) != word_bag(
            b["body"]
        ):
            raise ValueError("Scheme pair changed content or word frequencies")
        pairs.append(
            {
                "pair": len(pairs),
                **dict(zip(keys, key, strict=True)),
                "lo": a["unit"],
                "hi": b["unit"],
                "orientation": 2 * a["polarity"] - 1,
            }
        )
    return pairs


def incidence(history):
    """Return mention-to-event incidence and equal-event/mention weights for A."""
    validate_history(history)
    if any(r["actor"] != "A" for r in history):
        raise ValueError("This roster contains only A's action observations")
    events = sorted({r["event_id"] for r in history})
    b = np.array([[r["event_id"] == event for event in events] for r in history], float)
    count = b.sum(axis=0)
    weights = b @ (1 / count) / len(events)
    return events, b, weights, np.full(len(history), 1 / len(history))


def references(row, atoms, candidate_values):
    """Keep actual-code and code-neutral event references separate from mentions."""
    events, _, weights, mention = incidence(row["history"])
    laws, neutral = [], []
    for r in row["history"]:
        key = (row["content"], r["family"], r["polarity"])
        laws.append(atoms[(*key, r["occasion_code"])])
        neutral.append(np.mean([atoms[(*key, code)] for code in ORDERS[0]], axis=0))
    x, y, v = np.asarray(laws), np.asarray(neutral), np.asarray(candidate_values)
    if x.shape != (4, 10) or y.shape != (4, 10) or not np.isfinite(x).all():
        raise ValueError("All four actual-code atom laws required")
    values = {
        "event_actual_code": weights @ x,
        "event_code_neutral": weights @ y,
        "mention_actual_code": mention @ x,
    }
    return {
        "events": len(events),
        "mentions": len(row["history"]),
        "event_weights": weights,
        "mention_weights": mention,
        "laws": values,
        "scores": {k: float(p @ v) for k, p in values.items()},
    }


def contrast(low, high, orientation):
    """Keep scalar, log-odds and probability changes without a accuracy threshold."""
    a, b = np.asarray(low["probabilities"]), np.asarray(high["probabilities"])
    score, odds = high["score"] - low["score"], high["log_odds"] - low["log_odds"]
    return {
        "score": float(score),
        "oriented_score": float(score * orientation),
        "log_odds": float(odds),
        "oriented_log_odds": float(odds * orientation),
        "probability_L1_half": float(np.abs(b - a).sum() / 2),
    }


def character_components(values):
    """Exactly decompose the eight scope/syntax/order contrast cells, not neurons."""
    if set(values) != set(itertools.product((0, 1), repeat=3)):
        raise ValueError("All eight binary cells required")
    result = {}
    for mask in itertools.product((0, 1), repeat=3):
        result["".join(map(str, mask))] = (
            math.fsum(
                value
                * math.prod(
                    (-1) ** (m * bit) for m, bit in zip(mask, cell, strict=True)
                )
                for cell, value in values.items()
            )
            / 8
        )
    return result


def summarize(rows):
    """Report all fixed contrasts descriptively; templates are not independent people."""
    result = {"cases": len(rows)}
    for key in ("score", "oriented_score", "log_odds", "probability_L1_half"):
        x = np.asarray([r[key] for r in rows], float)
        result[key + "_mean"] = float(x.mean())
        result[key + "_rms"] = float(np.sqrt(np.mean(x * x)))
    result["oriented_positive"] = int(sum(r["oriented_score"] > 0 for r in rows))
    return result
