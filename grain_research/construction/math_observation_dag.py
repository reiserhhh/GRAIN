#!/usr/bin/env python3
"""Addressed proposition DAGs and purpose projections before aggregation.

All raw words and predicate nodes survive. A context predicate cannot acquire
the identity, actor or modality of its containing observation just because the
dependency parser attached it to that observation's verb. Linguistic decisions
are explicit adapters, not a learned universal semantic parser.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from grain_research.construction import math_purpose_accounts as P
from grain_research.source_event_graph import validate_words

ACCOUNTS = (*P.ACCOUNTS, "desire_about_other")
SELF = {"i", "me", "myself"}
COLLECTIVE = {"we", "us", "ourselves"}


def material_bank(source):
    """Create new object clauses and balanced modes before any parser readout."""
    actions = {
        (r["target"], r["action_index"]): r["action"]
        for r in source
        if r["object_index"] == 0 and r["bits"][1:] == [1, 1, 1]
    }
    objects = {
        "coordination_action": (
            "the volunteer teams preparing the riverside performance",
            "the local groups that are planning the community science fair",
        ),
        "verification_action": (
            "the numerical totals copied by the clerks from the delivery ledger",
            "the official figures that the bookkeepers recorded in the dispatch log",
        ),
    }
    rows = []
    for target, family, mode, style, polarity, own in product(
        sorted(objects), range(8), P.ACCOUNTS, range(2), (-1, 1), (-1, 1)
    ):
        action, obj = actions[target, family], objects[target][style]
        subject = "I" if own == 1 else "my colleague"
        do = "do" if own == 1 else "does"
        neg = " not" if polarity == -1 else ""
        if mode == "actual":
            text = f"At the rehearsal, {subject} did{neg} {action} {obj}."
        elif mode == "hypothetical":
            text = f"In a fictional account, {subject} would{neg} {action} {obj}."
        elif mode in {"desire", "plan"}:
            verb = "want" if mode == "desire" else "intend"
            pred = (
                f"{do} not {verb}"
                if polarity == -1
                else verb + ("s" if own == -1 else "")
            )
            text = f"{subject.capitalize()} {pred} to {action} {obj}."
        else:
            pred = (
                f"{do} not feel"
                if polarity == -1
                else "feel" + ("s" if own == -1 else "")
            )
            text = f"{subject.capitalize()} {pred} uneasy when {subject} must {action} {obj}."
        at = text.index(action + " " + obj)
        rows.append(
            {
                "material_id": f"new_{target}_{family}_{mode}_{style}_{polarity}_{own}",
                "target": target,
                "family": family,
                "style": style,
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
                        "bits": [1 if family < 4 else -1, polarity, own],
                    }
                ],
            }
        )
    by_key = {
        (
            r["target"],
            r["family"],
            r["events"][0]["mode"],
            r["style"],
            r["events"][0]["bits"][1],
            r["events"][0]["actor"],
        ): r
        for r in rows
    }
    for target, family, mode, swap in product(
        sorted(objects), range(4), ("actual", "desire", "state"), (False, True)
    ):
        a = by_key[target, family, mode, 0, 1, 1 if not swap else -1]
        b = by_key[
            target, (family + 1) % 4, "hypothetical", 1, -1, -1 if not swap else 1
        ]
        text = a["text"] + " " + b["text"]
        offset = len(a["text"]) + 1
        ea, eb = dict(a["events"][0]), dict(b["events"][0])
        ea["event_id"], eb["event_id"] = "a", "b"
        eb["span"] = [v + offset for v in eb["span"]]
        eb["action_span"] = [v + offset for v in eb["action_span"]]
        rows.append(
            {
                "material_id": f"mixed_{target}_{family}_{mode}_{int(swap)}",
                "target": target,
                "family": family,
                "style": 0,
                "kind": "mixture",
                "text": text,
                "events": [ea, eb],
            }
        )
    for target, family, case in product(sorted(objects), range(4), range(4)):
        action, obj = actions[target, family], objects[target][family % 2]
        texts = (
            f"I want my colleague to {action} {obj}.",
            f"My colleague wants me to {action} {obj}.",
            f"I imagine that I want to {action} {obj}.",
            f"I want not to {action} {obj}.",
        )
        text = texts[case]
        at = text.index(action + " " + obj)
        rows.append(
            {
                "material_id": f"scope_{target}_{family}_{case}",
                "target": target,
                "family": family,
                "style": case,
                "kind": "control" if case < 2 else "scope_challenge",
                "text": text,
                "events": [
                    {
                        "event_id": "e",
                        "span": [0, len(text)],
                        "action_span": [at, at + len(action)],
                        "action": action,
                        "object": obj,
                        "mode": "desire" if case < 2 else "unresolved",
                        "actor": -1 if case == 0 else 1,
                        "conceiver": -1 if case == 1 else 1,
                        "bits": [1, 1, -1 if case == 0 else 1],
                    }
                ],
            }
        )
    return rows


def observe_dag(sentences, text):
    """Build target-independent nodes, mode paths and actor/conceiver slots.

    Root/core complements are candidate observations. Non-core advcl/acl
    predicates remain context nodes with independent raw identities. A finite
    explicitly marked condition of a state predicate can supply its exposure
    context, but does not inherit that predicate's actor. Nested modes and lower
    negation remain relational objects, without an invented scalar reading.
    """
    nodes, edges, routed = [], [], []
    for sentence_index, words in enumerate(sentences):
        validate_words(words, text)
        by = {w["id"]: w for w in words}
        children = {i: [] for i in by}
        for w in words:
            if w["head"] in by:
                children[w["head"]].append(w["id"])

        def edge(i, by=by):
            return by[i]["deprel"].split(":")[0]

        def subtree(i, children=children):
            todo, seen = [i], set()
            while todo:
                j = todo.pop()
                if j not in seen:
                    seen.add(j)
                    todo.extend(children[j])
            return seen

        def subject(i, children=children):
            return [j for j in children[i] if edge(j) == "nsubj"]

        def spans(ids, by=by):
            return [
                [
                    min(by[j]["start"] for j in subtree(i)),
                    max(by[j]["end"] for j in subtree(i)),
                ]
                for i in ids
            ]

        def owner(ids, by=by):
            lemmas = {by[j]["lemma"] for j in ids}
            if len(ids) != 1:
                return "unknown"
            if lemmas & SELF:
                return "self"
            if lemmas & COLLECTIVE:
                return "collective"
            if by[ids[0]]["lemma"] in {"who", "which", "that"}:
                return "unresolved_relative"
            return "other"

        predicates = {
            w["id"]
            for w in words
            if w["upos"] == "VERB" or w["lemma"] in set(P.GOVERNORS) | P.UNSUPPORTED
        }
        modal_ids = {
            i
            for i in predicates
            if by[i]["lemma"] in P.GOVERNORS
            or (
                by[i]["lemma"] in P.UNSUPPORTED
                and any(edge(j) in {"ccomp", "xcomp"} for j in children[i])
            )
        }
        for w in words:
            if w["id"] not in predicates:
                continue
            i = w["id"]
            path, cur = [i], w
            while cur["head"]:
                cur = by[cur["head"]]
                path.append(cur["id"])
            operators = [j for j in path[1:] if by[j]["lemma"] in P.GOVERNORS]
            unsupported = [
                j
                for j in path[1:]
                if j in modal_ids and by[j]["lemma"] in P.UNSUPPORTED
            ]
            modal = i in modal_ids
            modifiers = any(edge(j) in {"acl", "amod"} for j in path)
            adjuncts = [j for j in path if edge(j) == "advcl"]
            head = operators[0] if operators else i
            marks = {
                by[j]["lemma"]
                for a in adjuncts
                for j in children[a]
                if edge(j) in {"mark", "advmod"}
            }
            state_condition = (
                bool(operators)
                and P.GOVERNORS[by[head]["lemma"]] == "state"
                and bool(subject(i))
                and bool(marks & {"when", "while", "whenever"})
            )
            if modal:
                node_type = "operator"
            elif modifiers:
                node_type = "object_context"
            elif adjuncts and not state_condition:
                node_type = "adjunct_context"
            elif state_condition:
                node_type = "state_condition"
            elif edge(i) in {"root", "conj"} or (
                operators
                and all(edge(j) in {"xcomp", "ccomp"} for j in path[: path.index(head)])
            ):
                node_type = "main_proposition"
            else:
                node_type = "embedded_proposition"
            actor, origin = subject(i), "direct"
            if not actor and edge(i) == "xcomp" and operators:
                controller = [j for j in children[head] if edge(j) in {"obj", "iobj"}]
                actor = controller or subject(head)
                origin = "object_control_candidate" if controller else "subject_control"
            if not actor and edge(i) == "conj" and by[i]["head"]:
                actor, origin = subject(by[i]["head"]), "conj_subject"
            conceiver = subject(head) if operators else actor
            mode_path = [P.GOVERNORS[by[j]["lemma"]] for j in reversed(operators)]
            mode = P.GOVERNORS[by[head]["lemma"]] if operators else "actual"
            if not operators and any(by[j]["lemma"] == "would" for j in children[i]):
                mode = "hypothetical"
            flags = []
            if len(operators) > 1 or unsupported:
                flags.append("nested_or_reported_scope")
            if (
                operators
                and mode in {"desire", "plan"}
                and any(by[j]["lemma"] in {"not", "never"} for j in children[i])
            ):
                flags.append("lower_clause_negation")
            if owner(actor) not in {"self", "other"} or owner(conceiver) not in {
                "self",
                "other",
            }:
                flags.append("unresolved_participant")
            if "nsubj:pass" in {by[j]["deprel"] for j in actor}:
                flags.append("passive_actor_not_patient")
            nid = f"s{sentence_index}:p{i}"
            parent = next((j for j in path[1:] if j in predicates), None)
            objects = [j for j in children[i] if edge(j) == "obj"]
            node = {
                "node_id": nid,
                "type": node_type,
                "action": w["lemma"],
                "action_span": [w["start"], w["end"]],
                "ud_path": path,
                "mode_path": mode_path,
                "candidate_mode": mode,
                "mode": mode if not flags else "unresolved",
                "status": "unresolved" if flags else "ok",
                "flags": flags,
                "actor": owner(actor),
                "conceiver": owner(conceiver),
                "actor_spans": spans(actor),
                "conceiver_spans": spans(conceiver),
                "subject_origin": origin,
                "sentence_index": sentence_index,
                "projection_text": text[
                    min(x["start"] for x in words) : max(x["end"] for x in words)
                ],
                "raw_object_spans": spans(objects),
                "root_event_identity_inherited": False,
            }
            nodes.append(node)
            if parent is not None:
                edges.append(
                    {
                        "parent": f"s{sentence_index}:p{parent}",
                        "child": nid,
                        "relation": edge(i),
                        "raw_path": path[: path.index(parent) + 1],
                    }
                )
            if node_type in {"main_proposition", "state_condition"}:
                for obj_span in spans(objects):
                    routed.append(
                        {
                            **node,
                            "object": text[obj_span[0] : obj_span[1]],
                            "object_span": obj_span,
                            "local_event_id": nid,
                        }
                    )
    return {
        "nodes": nodes,
        "edges": edges,
        "events": routed,
        "raw_text_retained": True,
        "raw_word_count": sum(map(len, sentences)),
    }


def readout(z, events, account, *, typed_owner):
    """Project typed evidence; preserve other-directed desires as a separate unit."""
    z = np.array(z, dtype=float, copy=True).reshape(-1, 4)
    if len(z) != len(events) or not np.isfinite(z).all():
        raise ValueError("invalid observation coordinates")
    roles = []
    modes = []
    for e in events:
        actor, conceiver, mode = e.get("actor"), e.get("conceiver"), e["mode"]
        if account == "desire_about_other":
            roles.append(conceiver == "self" and actor == "other")
            modes.append("desire" if mode == "desire" else "unresolved")
        else:
            roles.append(
                conceiver == "self"
                if account == "state"
                else actor == "self"
                and (conceiver == "self" or mode in {"actual", "hypothetical"})
            )
            modes.append(mode)
    effective = "desire" if account == "desire_about_other" else account
    if typed_owner:
        z[:, 2] = np.where(roles, 1.0, -1.0)
    return P.purpose_values(z, modes, effective)


def compression_witness():
    """Exhibit a kernel direction lost by untyped aggregation but not scoring."""
    aggregation = np.array([[1.0, 1.0]])
    route = np.array([[1.0, 0.0]])
    null_direction = np.array([1.0, -1.0])
    return {
        "aggregation": aggregation.tolist(),
        "purpose_projection": route.tolist(),
        "null_direction": null_direction.tolist(),
        "aggregate_difference": float((aggregation @ null_direction)[0]),
        "routed_difference": float((route @ null_direction)[0]),
        "criterion": "F factors through S iff ker(S) is contained in ker(F)",
    }


def transported_readout(values, projection, change):
    """Transport both typed projection and its dual; a fixed mask is not invariant."""
    values, projection, change = (
        np.asarray(x, float) for x in (values, projection, change)
    )
    n = len(values)
    if projection.shape != (n, n) or change.shape != (n, n):
        raise ValueError("observation transport shape")
    new_values = np.linalg.solve(change, values)
    new_projection = np.linalg.solve(change, projection @ change)
    dual = np.ones(n) @ change
    return {
        "original": float(np.ones(n) @ projection @ values),
        "transported": float(dual @ new_projection @ new_values),
        "fixed_mask": float(np.ones(n) @ projection @ new_values),
        "projection_idempotence_error": float(
            np.max(np.abs(new_projection @ new_projection - new_projection))
        ),
    }


def replay_graph_records(records):
    """Replay supplied record versions, invalidating all dependent node versions.

    A context node is linked to its container record, not declared to be the
    same physical event as its main proposition. Aliases add no independent
    evidence; a revision replaces both main and context snapshots together.
    """
    active = P.replay_versions(records)
    nodes = []
    for record in active:
        graph = record["graph"]
        ids = [n["node_id"] for n in graph["nodes"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate local node identity")
        for node in graph["nodes"]:
            nodes.append(
                {
                    **node,
                    "node_id": record["record_id"] + "/" + node["node_id"],
                    "record_id": record["record_id"],
                    "container_event_id": record["event_id"],
                    "author_id": record["author_id"],
                }
            )
    return active, nodes
