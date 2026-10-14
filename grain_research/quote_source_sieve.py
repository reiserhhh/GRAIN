#!/usr/bin/env python3
"""Resolve explicit reporting sources before a fallible generic NLI fallback."""

from __future__ import annotations

from grain_research.source_event_graph import REPORT, event_graph, quote_spans


def explicit_quote_source(text, sentences, start, end, fallback):
    """Bind a quoted passage to a reporting source, preserving uncertainty.

    Dependency-governing report verbs take precedence over a same-sentence
    nearest preceding cue. This is a source-reference heuristic, not factuality
    inference or general coreference resolution.
    """
    if not 0 <= start < end <= len(text):
        raise ValueError("Valid quote address required")
    all_words = [w for sentence in sentences for w in sentence]
    quotes = quote_spans(all_words)
    first = next(
        (s for s in sentences if any(start <= w["start"] < end for w in s)), None
    )
    if first is None:
        raise ValueError("Quote has no parsed words")
    events = event_graph(first, text, quote_context=quotes)
    by_id = {e["head_id"]: e for e in events}
    words = {w["id"]: w for w in first}
    outside = {
        e["head_id"]: e
        for e in events
        if e["predicate"] in REPORT
        and not any(
            q["balanced"] and q["start"] <= e["predicate_start"] < q["end"]
            for q in quotes
        )
    }
    connected = []
    for event in events:
        if start <= event["predicate_start"] < end:
            connected.extend(
                p["predicate_id"]
                for p in event["embedding_path"]
                if p["predicate_id"] in outside
            )
    if connected:
        cue_id = min(
            set(connected), key=lambda i: abs(by_id[i]["predicate_start"] - start)
        )
        mode = "dependency_report_source"
    else:
        before = [e for e in outside.values() if e["predicate_end"] <= start]
        if not before:
            return {
                "source_owner": fallback["source_owner"],
                "mechanism": "nli_fallback",
                "reason": fallback["reason"],
                "cue_span": None,
                "source_spans": [],
            }
        cue_id = max(before, key=lambda e: e["predicate_end"])["head_id"]
        mode = "same_sentence_preceding_report_source"
    cue = by_id[cue_id]
    route = "unresolved"
    reason = "unresolved_reporting_scope"
    if not (
        cue["local_negation"]
        or cue["modal_cues"]
        or cue["source_path"]
        or cue["passive_subject"]
        or cue["subject_origin"] != "direct"
    ):
        if cue["participant_kind"] == "first_person_singular":
            route, reason = "self", "explicit_writer_reporting_subject"
        elif cue["participant_kind"] == "other_or_generic":
            route, reason = "other", "explicit_other_reporting_subject"
    return {
        "source_owner": route,
        "mechanism": mode,
        "reason": reason,
        "cue_span": [cue["predicate_start"], cue["predicate_end"]],
        "source_spans": [
            [words[i]["start"], words[i]["end"]] for i in cue["subject_ids"]
        ],
        "source_origin": cue["subject_origin"],
        "cue": cue["predicate"],
    }
