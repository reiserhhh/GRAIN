#!/usr/bin/env python3
"""Event-local role projections and composition witnesses, without target lookup.

The projection is a scoped UD dependency interface, not a full natural-language
semantics parser. Missing scopes are kept explicit. No true material roles are
accepted by the projection or by the cold scoring functions.
"""

from __future__ import annotations

import hashlib
import re
from itertools import product

import numpy as np

from grain_research.source_event_graph import validate_words


def compose(events, style):
    """Join supplied event records, preserving explicit identity and raw spans."""
    text, addressed = "", []
    for j, event in enumerate(events):
        part = event["text"].rstrip(".")
        if j:
            text += ". " if style == "sentences" else ", but "
            if style == "conjoined":
                part = part[0].lower() + part[1:]
        start = len(text)
        text += part
        addressed.append({**event, "span": [start, len(text)]})
    return text + ".", addressed


def material_bank(source, styles=("sentences", "conjoined")):
    """Create new full role mixtures using only source-packet cold object slots."""
    lookup = {
        (r["target"], r["action_index"], r["object_index"], *r["bits"][1:]): i
        for i, r in enumerate(source)
    }
    rows = []

    def event(target, ai, oi, roles, identity):
        i = lookup[(target, ai, oi, *roles)]
        original = source[i]
        if original["role"] != "unseen_object":
            raise ValueError("new role packet must retain source-only semantic fits")
        return {
            "event_id": identity,
            "source_index": i,
            "text": original["text"],
            "bits": original["bits"],
        }

    def row(target, family, style, roles, events, kind, suffix):
        text, addressed = compose(events, style)
        key = f"{target}_{family}_{style}_{suffix}"
        return {
            "material_id": key,
            "target": target,
            "family": family,
            "style": style,
            "kind": kind,
            "roles": list(roles),
            "events": addressed,
            "text": text,
            "ambiguous": False,
        }

    role_cells = list(product((-1, 1), repeat=3))
    for target in sorted({r["target"] for r in source}):
        for f in range(8):
            ai, oi = f % 4, 8 + f % 2
            bi, bo = (f + 1) % 4, (8 if f < 4 else 10) + (f + 1) % 2
            for ra, rb, style in product(role_cells, role_cells, styles):
                suffix = "_".join(str(v) for v in (*ra, *rb))
                ea, eb = event(target, ai, oi, ra, "a"), event(target, bi, bo, rb, "b")
                base = row(target, f, style, (*ra, *rb), [ea, eb], "mixture", suffix)
                rows.append(base)
                # Explicit aliases are supplied record relations, not inferred from words.
                if ra[1:] == (1, 1) and rb[1:] == (-1, 1):
                    alias = row(
                        target,
                        f,
                        style,
                        (*ra, *rb),
                        [ea, eb, ea],
                        "alias",
                        suffix + "_alias",
                    )
                    added = event(target, (f + 2) % 4, 8 + (f + 1) % 2, (1, 1, 1), "c")
                    novel = row(
                        target,
                        f,
                        style,
                        (*ra, *rb),
                        [ea, eb, added],
                        "new_event",
                        suffix + "_new",
                    )
                    alias["base_id"] = novel["base_id"] = base["material_id"]
                    rows.extend((alias, novel))
            for case in (
                "shared_positive",
                "local_negative",
                "shared_modality",
                "ambiguous_negation",
            ):
                ra = (
                    (-1, 1, 1)
                    if case == "ambiguous_negation"
                    else (1, 1, -1)
                    if case == "shared_modality"
                    else (1, 1, 1)
                )
                rb = (-1, 1, 1) if case == "local_negative" else (1, 1, ra[2])
                ea, eb = event(target, ai, oi, ra, "a"), event(target, bi, bo, rb, "b")
                b = source[eb["source_index"]]
                fragment = (
                    ("did not " if case == "local_negative" else "")
                    + b["action"]
                    + " "
                    + b["object"]
                )
                first = ea["text"].rstrip(".")
                text = first + " and " + fragment + "."
                record = row(target, f, "inherited", (*ra, *rb), [ea, eb], case, case)
                record["text"] = text
                record["events"][0]["span"] = [0, len(first)]
                record["events"][1]["span"] = [len(first) + 5, len(text) - 1]
                if case == "ambiguous_negation":
                    record["ambiguous"] = True
                    record["alternative_second_polarities"] = [-1, 1]
                rows.append(record)
    return rows


def project_events(sentences, text):
    """Project main/conjoined clauses; bind raw role addresses and provenance.

    Explicit roles override inherited roles. Only a missing subject permits
    propagation from a coordination head. A bare second verb under a negative
    head is flagged ambiguous instead of assigning a guessed scope.
    """
    projected = []
    for sentence_index, words in enumerate(sentences):
        validate_words(words, text)
        by_id = {w["id"]: w for w in words}
        children = {i: [] for i in by_id}
        for w in words:
            if w["head"] in children:
                children[w["head"]].append(w["id"])

        def edge(i, by_id=by_id):
            return by_id[i]["deprel"].split(":")[0]

        def branch(i, children=children):
            ids, queue = set(), [i]
            while queue:
                j = queue.pop()
                if j not in ids:
                    ids.add(j)
                    queue.extend(children[j])
            return ids

        heads = {
            w["id"] for w in words if w["upos"] == "VERB" and edge(w["id"]) == "root"
        }
        while True:
            expanded = heads | {
                w["id"]
                for w in words
                if w["upos"] == "VERB"
                and edge(w["id"]) in {"conj", "parataxis"}
                and w["head"] in heads
            }
            if expanded == heads:
                break
            heads = expanded
        for head in sorted(heads, key=lambda i: by_id[i]["start"]):
            objects = [i for i in children[head] if edge(i) == "obj"]
            for obj in objects:
                ids = {head}
                for child in children[head]:
                    if edge(child) not in {
                        "conj",
                        "parataxis",
                        "cc",
                        "advcl",
                        "ccomp",
                        "xcomp",
                    }:
                        ids |= branch(child)
                # Clause boundaries are pruned from the predicate, not from its object subtree.
                object_ids = branch(obj)
                start, end = (
                    min(by_id[i]["start"] for i in object_ids),
                    max(by_id[i]["end"] for i in object_ids),
                )
                subject = [i for i in children[head] if edge(i) in {"nsubj", "csubj"}]
                auxiliaries = [i for i in children[head] if edge(i) in {"aux", "cop"}]
                subject_origin, auxiliary_origin, frame_origin = (
                    "direct",
                    "direct",
                    "direct",
                )
                flags, inherited = [], {}
                parent = by_id[head]["head"]
                if not subject and edge(head) == "conj" and parent in heads:
                    subject = [
                        i for i in children[parent] if edge(i) in {"nsubj", "csubj"}
                    ]
                    if subject:
                        subject_origin = "coordination_head"
                        inherited["subject"] = subject
                        for i in subject:
                            ids |= branch(i)
                        if not auxiliaries:
                            auxiliaries = [
                                i for i in children[parent] if edge(i) in {"aux", "cop"}
                            ]
                            auxiliary_origin = "coordination_head"
                            inherited["auxiliary"] = auxiliaries
                            ids |= set(auxiliaries)
                            if any(
                                by_id[i]["lemma"] in {"not", "never"}
                                for i in children[parent]
                            ):
                                flags.append("ambiguous_shared_negation")
                        frame = [
                            i
                            for i in children[parent]
                            if edge(i) in {"obl", "advmod"}
                            and by_id[i]["start"]
                            < min(by_id[s]["start"] for s in subject)
                        ]
                        if frame:
                            frame_origin = "coordination_head"
                            inherited["frame"] = frame
                            for i in frame:
                                ids |= branch(i)
                            first_subject = min(by_id[s]["start"] for s in subject)
                            ids |= {
                                i
                                for i in children[parent]
                                if edge(i) == "punct"
                                and by_id[i]["start"] < first_subject
                            }
                if not subject:
                    flags.append("missing_subject")
                if not auxiliaries and "VerbForm=Fin" not in by_id[head]["feats"]:
                    flags.append("missing_finite_operator")
                selected = sorted(
                    (by_id[i] for i in ids if edge(i) != "cc"),
                    key=lambda w: (w["start"], w["end"]),
                )
                content = [w for w in selected if w["upos"] != "PUNCT"]
                if not content:
                    continue
                limit = max(w["end"] for w in content)
                pieces, addresses = [], set()
                for w in selected:
                    if w["upos"] == "PUNCT" and w["start"] >= limit:
                        continue
                    key = (w["start"], w["end"])
                    if key not in addresses:
                        pieces.append(
                            {
                                "span": list(key),
                                "surface": text[key[0] : key[1]],
                                "word_ids": [w["id"]],
                            }
                        )
                        addresses.add(key)
                projection = (
                    re.sub(
                        r"\s+([,.;!?])", r"\1", " ".join(p["surface"] for p in pieces)
                    ).rstrip(".,; ")
                    + "."
                )
                projected.append(
                    {
                        "sentence_index": sentence_index,
                        "head_id": head,
                        "action": by_id[head]["lemma"],
                        "action_span": [by_id[head]["start"], by_id[head]["end"]],
                        "object": text[start:end],
                        "object_span": [start, end],
                        "projection_text": projection,
                        "pieces": pieces,
                        "subject_origin": subject_origin,
                        "auxiliary_origin": auxiliary_origin,
                        "frame_origin": frame_origin,
                        "inherited_word_ids": inherited,
                        "status": "unresolved" if flags else "ok",
                        "flags": flags,
                    }
                )
    return projected


def attach_event_identity(projected, supplied):
    """Attach supplied record identity by head address, never by a true role label."""
    result = []
    for e in projected:
        candidates = [
            (i, r)
            for i, r in enumerate(supplied)
            if r["span"][0] <= e["action_span"][0] < r["span"][1]
        ]
        if len(candidates) != 1:
            result.append(
                {
                    **e,
                    "event_id": None,
                    "record_index": None,
                    "status": "unresolved",
                    "flags": e["flags"] + ["unbound_record_identity"],
                }
            )
        else:
            index, record = candidates[0]
            result.append({**e, "event_id": record["event_id"], "record_index": index})
    return result


def missing_records(events, expected_count):
    """Count gaps or multiplicity per addressed record, not just total heads."""
    counts = [0] * expected_count
    failures = 0
    for event in events:
        index = event.get("record_index")
        if event["status"] != "ok" or index is None or not 0 <= index < expected_count:
            failures += 1
        else:
            counts[index] += 1
    return failures + sum(abs(n - 1) for n in counts)


def event_profile(events, scores, qualified, *, unresolved=0):
    """Keep one supplied observation per event; report alias variability separately.

    The first record is retained before qualification. Repeated identities are
    not extra information. Missing events do not become zero-valued observations.
    """
    first, variation = {}, []
    for event, score, q in zip(events, scores, qualified, strict=True):
        identity = event["event_id"]
        if identity is None:
            raise ValueError("scoring requires supplied record identity")
        state = (float(score), bool(q))
        if identity in first:
            variation.append(abs(state[0] - first[identity][0]))
        else:
            first[identity] = state
    observed = [value for value, q in first.values() if q]
    return {
        "score": float(np.mean(observed)) if observed and not unresolved else None,
        "known_score": float(np.mean(observed)) if observed else None,
        "observed_events": len(observed),
        "unique_events": len(first),
        "unit": float(np.mean([v for v, _ in first.values()]))
        if first and not unresolved
        else None,
        "complete": unresolved == 0,
        "unresolved": unresolved,
        "alias_score_variation": max(variation, default=0.0),
    }


def true_profile(row):
    """Evaluation-only material contract; never used by the cold observer."""
    events, scores, qualified = [], [], []
    for e in row["events"]:
        k, p, o, a = e["bits"]
        q = k == o == a == 1
        events.append({"event_id": e["event_id"]})
        scores.append(float(p if q else 0))
        qualified.append(q)
    return event_profile(events, np.array(scores), np.array(qualified))


def marginal_collision_witness():
    """Equal role marginals lose different local person-polarity associations."""
    # k,p,o,a; every slot has the same marginal distribution in both worlds.
    left = np.array([[1, -1, 1, 1], [1, 1, -1, 1]])
    right = np.array([[1, 1, 1, 1], [1, -1, -1, 1]])

    def observed(bits):
        q = np.all(bits[:, [0, 2, 3]] == 1, axis=1)
        return float(bits[q, 1].mean())

    return {
        "left": left.tolist(),
        "right": right.tolist(),
        "marginals_equal": bool(np.array_equal(left.mean(axis=0), right.mean(axis=0))),
        "left_author_expression": observed(left),
        "right_author_expression": observed(right),
        "joint_po_moment": [float(np.mean(v[:, 1] * v[:, 2])) for v in (left, right)],
        "scope": "known role-composition witness, not a psychological theorem",
    }


def text_digest(text):
    """Bind caches and observation projections to exact public source text."""
    return hashlib.sha256(text.encode()).hexdigest()
