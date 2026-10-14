#!/usr/bin/env python3
"""Address-preserving query/event views; extraction is not logical entailment."""

from __future__ import annotations

import numpy as np

from grain_research.source_event_graph import CLAUSE_EDGES, validate_words


def render_fragments(text, words, selected):
    """Render exact selected surface spans with explicit discontinuity markers."""
    by_id = {w["id"]: w for w in words}
    if not selected or not set(selected) <= by_id.keys():
        raise ValueError("Nonempty known word IDs required")
    spans = sorted({(by_id[i]["start"], by_id[i]["end"]) for i in selected})
    segments = []
    for start, end in spans:
        if segments and (
            start <= segments[-1][1] or text[segments[-1][1] : start].isspace()
        ):
            segments[-1][1] = max(end, segments[-1][1])
        else:
            segments.append([start, end])
    return " ... ".join(text[a:b] for a, b in segments), segments


def event_views(text, words, event, quotes=()):
    """Return local fragments and an original-text scope envelope.

    The envelope retains syntactic ancestry and connected adverbial clauses.
    Neither view resolves all pronouns or establishes whether an event happened.
    """
    validate_words(words, text)
    by_id = {w["id"]: w for w in words}
    children = {i: [] for i in by_id}
    for w in words:
        if w["head"]:
            children[w["head"]].append(w["id"])

    def branch(i, stop):
        ids = {i}
        for child in children[i]:
            edge = by_id[child]["deprel"].split(":")[0]
            if not stop or edge not in CLAUSE_EDGES:
                ids.update(branch(child, stop))
        return ids

    local_ids = set(event["local_word_ids"]) | set(event["subject_ids"])
    local_text, local_segments = render_fragments(text, words, local_ids)
    scope_ids = set(local_ids)
    path_ids = [event["head_id"]] + [e["predicate_id"] for e in event["embedding_path"]]
    for i in path_ids:
        scope_ids.update(branch(i, True))
        for c in children[i]:
            if by_id[c]["deprel"].split(":")[0] == "advcl":
                scope_ids.update(branch(c, False))
    for p in event["embedding_path"]:
        scope_ids.update(p["subject_ids"])
        scope_ids.update(p["source_ids"])
    start = min(by_id[i]["start"] for i in scope_ids)
    end = max(by_id[i]["end"] for i in scope_ids)
    for q in quotes:
        if q["balanced"] and q["start"] <= event["predicate_start"] < q["end"]:
            start, end = min(start, q["start"]), max(end, q["end"])
    if not 0 <= start < end <= len(text):
        raise ValueError("Invalid scope envelope")
    return {
        "local_text": local_text,
        "local_segments": local_segments,
        "local_ids": sorted(local_ids),
        "scope_text": text[start:end],
        "scope_start": start,
        "scope_end": end,
        "scope_ids": sorted(scope_ids),
        "scope_equals_local": text[start:end] == local_text,
        "local_is_fragment": len(local_segments) > 1,
        "scope_extra_words": sum(
            start <= w["start"] < end and w["id"] not in scope_ids for w in words
        ),
    }


def rank_events(event_ids, scores):
    """Deterministic cosine ranking and explicit near-tie count, no threshold fit."""
    values = np.asarray(scores, float)
    if (
        values.shape != (len(event_ids),)
        or not len(values)
        or not np.isfinite(values).all()
        or len(set(event_ids)) != len(event_ids)
    ):
        raise ValueError("Unique events and finite one-dimensional scores required")
    order = sorted(range(len(values)), key=lambda i: (-values[i], event_ids[i]))
    top = order[0]
    return {
        "index": top,
        "score": float(values[top]),
        "margin": float(values[top] - values[order[1]]) if len(order) > 1 else None,
        "near_ties": int(np.sum(np.abs(values - values[top]) <= 1e-7)),
    }


def evidence_route(event):
    """Route grammatical evidence without relabeling participant as semantic agent."""
    subject = event["subject_reference"]
    if subject in {"unresolved", "mixed", "quoted_first_person_unresolved"}:
        route = "unresolved_participant"
    elif subject != "writer_first_person_candidate":
        route = "other_collective_or_quoted_context"
    elif event["source_path"]:
        route = "writer_subject_embedded_source"
    else:
        route = "writer_subject_candidate"
    return {
        "route": route,
        "subject_reference": subject,
        "subject_origin": event["subject_origin"],
        "source_candidate": event["source_candidate"],
        "passive_subject": event["passive_subject"],
        "intention_embedded": any(p["intention"] for p in event["embedding_path"]),
        "ancestor_negation": any(p["local_negation"] for p in event["embedding_path"]),
        "local_negation": bool(event["local_negation"]),
        "modal": bool(event["modal_cues"]),
        "adverbial_linked": bool(event["adverbial_clauses"]),
        "factuality": "not_inferred",
    }
