#!/usr/bin/env python3
"""Finite object/context views, without transferring context event ownership.

An attachment candidate is an observation hypothesis, not a second event or a
confidence interval. Its envelope records sensitivity to the observation law.
The frozen vector function is evaluated only after these views are constructed.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from grain_research.construction import math_purpose_accounts as P
from grain_research.source_event_graph import validate_words


def material_bank(source):
    """Make cold modifier/adjunct contrasts before parsing or vector scoring."""
    actions = {
        (r["target"], r["action_index"]): r["action"]
        for r in source
        if r["object_index"] == 0 and r["bits"][1:] == [1, 1, 1]
    }
    objects = {
        "coordination_action": (
            (
                "the volunteer crews",
                "preparing the winter benefit concert",
                "that are preparing the winter benefit concert",
                "the printed portraits",
                "showing the volunteer crews",
            ),
            (
                "the local groups",
                "planning the coastal science exhibit",
                "that are planning the coastal science exhibit",
                "the decorative namecards",
                "depicting the local groups",
            ),
        ),
        "verification_action": (
            (
                "the shipment figures",
                "copied from the warehouse dispatch records",
                "that the clerks copied from the warehouse dispatch records",
                "the ornamental lettering",
                "covering the shipment brochure",
            ),
            (
                "the financial totals",
                "recorded in the original expense ledger",
                "that the accountants recorded in the original expense ledger",
                "the colored illustrations",
                "decorating the financial booklet",
            ),
        ),
    }
    variants = (
        "reduced_positive",
        "finite_positive",
        "reduced_negative",
        "other_adjunct",
    )
    rows = []
    for target, family, object_family, variant, mode, polarity, own in product(
        sorted(objects), range(8), range(2), variants, P.ACCOUNTS, (-1, 1), (-1, 1)
    ):
        action = actions[target, family]
        base, reduced, finite, bad, bad_context = objects[target][object_family]
        suffix = ""
        if variant == "reduced_positive":
            obj = base + " " + reduced
        elif variant == "finite_positive":
            obj = base + " " + finite
        elif variant == "reduced_negative":
            obj = bad + " " + bad_context
        else:
            obj = bad
            other_action = "organizes" if target == "coordination_action" else "checks"
            suffix = f" while my colleague {other_action} {base} {reduced}"
        subject = "I" if own == 1 else "my colleague"
        do = "do" if own == 1 else "does"
        neg = " not" if polarity == -1 else ""
        if mode == "actual":
            text = f"At this meeting, {subject} did{neg} {action} {obj}{suffix}."
        elif mode == "hypothetical":
            text = (
                f"In a fictional account, {subject} would{neg} {action} {obj}{suffix}."
            )
        elif mode in {"desire", "plan"}:
            verb = "want" if mode == "desire" else "intend"
            pred = (
                f"{do} not {verb}"
                if polarity == -1
                else verb + ("s" if own == -1 else "")
            )
            text = f"{subject.capitalize()} {pred} to {action} {obj}{suffix}."
        else:
            pred = (
                f"{do} not feel"
                if polarity == -1
                else "feel" + ("s" if own == -1 else "")
            )
            text = f"{subject.capitalize()} {pred} uneasy when {subject} must {action} {obj}{suffix}."
        at = text.index(action + " " + obj)
        rows.append(
            {
                "material_id": f"joint_{target}_{family}_{object_family}_{variant}_{mode}_{polarity}_{own}",
                "target": target,
                "family": family,
                "object_family": object_family,
                "variant": variant,
                "kind": "single",
                "text": text,
                "events": [
                    {
                        "event_id": "e",
                        "span": [0, len(text)],
                        "action_span": [at, at + len(action)],
                        "action": action,
                        "object": obj,
                        "mode": mode,
                        "actor": own,
                        "conceiver": own,
                        "bits": [
                            1 if family < 4 and variant.endswith("positive") else -1,
                            polarity,
                            own,
                        ],
                    }
                ],
            }
        )
    # A comma-marked unassigned participle is deliberately not an object modifier.
    # It remains context, including its ambiguity, rather than another author act.
    for target, family, object_family, own in product(
        sorted(objects), range(8), range(2), (-1, 1)
    ):
        action = actions[target, family]
        base, reduced, _, bad, _ = objects[target][object_family]
        subject = "I" if own == 1 else "my colleague"
        text = f"{subject.capitalize()} did {action} {bad}, preparing {base} for a public event."
        at = text.index(action + " " + bad)
        rows.append(
            {
                "material_id": f"comma_{target}_{family}_{object_family}_{own}",
                "target": target,
                "family": family,
                "object_family": object_family,
                "variant": "comma_adjunct",
                "kind": "attachment_challenge",
                "text": text,
                "events": [
                    {
                        "event_id": "e",
                        "span": [0, len(text)],
                        "action_span": [at, at + len(action)],
                        "action": action,
                        "object": bad,
                        "mode": "actual",
                        "actor": own,
                        "conceiver": own,
                        "bits": [-1, 1, own],
                    }
                ],
            }
        )
    for target, family, object_family, reverse in product(
        sorted(objects), range(4), range(2), (False, True)
    ):
        action = actions[target, family]
        base, reduced, _, _, _ = objects[target][object_family]
        obj = base + " " + reduced
        text = (
            f"My colleague wants me to {action} {obj}."
            if reverse
            else f"I want my colleague to {action} {obj}."
        )
        at = text.index(action + " " + obj)
        rows.append(
            {
                "material_id": f"control_{target}_{family}_{object_family}_{int(reverse)}",
                "target": target,
                "family": family,
                "object_family": object_family,
                "variant": "object_control",
                "kind": "control",
                "text": text,
                "events": [
                    {
                        "event_id": "e",
                        "span": [0, len(text)],
                        "action_span": [at, at + len(action)],
                        "action": action,
                        "object": obj,
                        "mode": "desire",
                        "actor": 1 if reverse else -1,
                        "conceiver": -1 if reverse else 1,
                        "bits": [1, 1, 1 if reverse else -1],
                    }
                ],
            }
        )
    return rows


def object_views(sentences, text, graph):
    """Observe addressed strict, legal-candidate and naive all-context views.

    Legal extensions require a directly attached, nonfinite unmarked predicate,
    no explicit subject/auxiliary, no comma boundary and no intervening main
    predicate. This yields candidates, not a guarantee of semantic attachment.
    Explicit-subject or marked adjuncts survive as context but cannot extend the
    object. Neither view construction nor candidate order uses target labels.
    """
    for words in sentences:
        validate_words(words, text)
    nodes = {n["node_id"]: n for n in graph["nodes"]}
    children = {}
    for edge in graph["edges"]:
        children.setdefault(edge["parent"], []).append(edge["child"])
    result = []
    for event in graph["events"]:
        words = sentences[event["sentence_index"]]
        by = {w["id"]: w for w in words}
        deps = {i: [] for i in by}
        for w in words:
            if w["head"] in by:
                deps[w["head"]].append(w["id"])

        def subtree(i, deps=deps):
            todo, seen = [i], set()
            while todo:
                j = todo.pop()
                if j not in seen:
                    seen.add(j)
                    todo.extend(deps[j])
            return seen

        start, end = event["object_span"]
        candidates = [
            {
                "kind": "strict",
                "span": [start, end],
                "text": text[start:end],
                "context_node_ids": [],
            }
        ]
        contexts, possible_ends, naive_ends = [], [], [end]
        for nid in sorted(children.get(event["node_id"], [])):
            n = nodes[nid]
            if n["type"] not in {
                "object_context",
                "adjunct_context",
                "embedded_proposition",
            }:
                continue
            i = n["ud_path"][0]
            ids = subtree(i)
            a, b = min(by[j]["start"] for j in ids), max(by[j]["end"] for j in ids)
            direct = [by[j] for j in deps[i]]
            blocked = [
                w["deprel"]
                for w in direct
                if w["deprel"].split(":")[0] in {"nsubj", "aux", "mark"}
            ]
            has_comma = "," in text[end:a] or any(w["surface"] == "," for w in direct)
            participle = "VerbForm=Part" in by[i].get("feats", "") or by[i][
                "surface"
            ].endswith("ing")
            legal = (
                n["type"] == "adjunct_context"
                and a >= end
                and not blocked
                and not has_comma
                and participle
            )
            contexts.append(
                {
                    "node_id": nid,
                    "span": [a, b],
                    "text": text[a:b],
                    "actor": n["actor"],
                    "type": n["type"],
                    "legal_extension": legal,
                    "blocked_dependencies": blocked,
                    "comma_boundary": has_comma,
                    "event_identity_inherited": False,
                }
            )
            if b > end:
                naive_ends.append(b)
            if legal:
                possible_ends.append((b, nid))
        for b, nid in sorted(possible_ends):
            candidates.append(
                {
                    "kind": "attachment_candidate",
                    "span": [start, b],
                    "text": text[start:b],
                    "context_node_ids": [nid],
                }
            )
        unique = {tuple(c["span"]): c for c in candidates}
        candidates = [unique[k] for k in sorted(unique)]
        legal_choice = candidates[-1]
        result.append(
            {
                **event,
                "candidates": candidates,
                "contexts": contexts,
                "legal_object": legal_choice["text"],
                "naive_object": text[start : max(naive_ends)],
                "has_attachment_alternative": len(candidates) > 1,
                "selected_by": "longest legal raw span, never target score",
            }
        )
    return result


def envelope(values):
    """Permutation-invariant range of finite observation candidates, not a CI."""
    x = np.asarray(values, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty candidate scores required")
    lo, hi = float(x.min()), float(x.max())
    return {
        "low": lo,
        "high": hi,
        "width": hi - lo,
        "all_positive": lo > 0,
        "any_positive": hi > 0,
        "gate_disagreement": lo <= 0 < hi,
    }


def cross_difference(f00, f01, f10, f11):
    """Four-corner action/context interaction in fixed vector-function units."""
    return np.asarray(f11) - np.asarray(f10) - np.asarray(f01) + np.asarray(f00)


def kernel_factorization_witness():
    """Show that discarding relation type can alias different required readouts."""
    collapse = np.array([[1.0, 1.0]])
    action_projection = np.array([[1.0, 0.0]])
    v = np.array([1.0, -1.0])
    return {
        "collapse": collapse.tolist(),
        "projection": action_projection.tolist(),
        "lost_direction": v.tolist(),
        "collapsed": float((collapse @ v)[0]),
        "qualified": float((action_projection @ v)[0]),
        "claim": "relation-blind aggregation need not factor a typed function",
    }
