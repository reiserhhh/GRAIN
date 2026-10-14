#!/usr/bin/env python3
"""Alternative effective parents for ambiguous state-clause coordination.

A basic UD root-conj can mean a second main proposition or a shared temporal
condition. Keep both scopes rather than choosing the one with a high score.
"""

from __future__ import annotations


def alternatives(data):
    """Return outside/inside candidate scopes using only actual words and DAG.

    The inside interpretation is a structural candidate, not a resolved parse.
    For a subjectless conjunct its actor can differ from the root experiencer:
    candidate temporal inheritance comes from the first condition actor.
    """
    results = []
    for report in data["reports"]:
        root = next(
            n for n in data["graph"]["nodes"] if n["node_id"] == report["report_id"]
        )
        si, rid = root["sentence_index"], root["ud_path"][0]
        words = {w["id"]: w for w in data["words"][si]}
        candidate = []
        already = {t["node_id"] for t in report["triggers"]}
        for n in data["graph"]["nodes"]:
            if (
                n["sentence_index"] != si
                or n["node_id"] in already
                or n["action"] == "feel"
            ):
                continue
            w = words[n["ud_path"][0]]
            if (
                w["head"] != rid
                or w["deprel"].split(":")[0] != "conj"
                or not n["raw_object_spans"]
                or not report["triggers"]
            ):
                continue
            actor = n["actor"]
            origin = n["subject_origin"]
            if origin == "conj_subject":
                actors = {t["actor"] for t in report["triggers"]}
                actor = next(iter(actors)) if len(actors) == 1 else "unknown"
                origin = "candidate_condition_subject"
            negative = any(
                x["head"] == w["id"] and x["lemma"] in {"not", "never"}
                for x in words.values()
            )
            for span in n["raw_object_spans"]:
                candidate.append(
                    {
                        "node_id": n["node_id"],
                        "report_id": report["report_id"],
                        "action": n["action"],
                        "action_span": n["action_span"],
                        "object": _span_text(data, span),
                        "object_span": span,
                        "actor": actor,
                        "subject_origin": origin,
                        "polarity": -1 if negative else 1,
                        "status": "ok" if actor in {"self", "other"} else "unresolved",
                        "scope": "possible_shared_temporal_condition_not_resolved",
                        "experiencer_inherited_as_actor": False,
                    }
                )
        results.append(
            {
                "report_id": report["report_id"],
                "outside": report,
                "inside_candidate": {
                    **report,
                    "triggers": report["triggers"] + candidate,
                    "status": "unresolved_scope" if candidate else report["status"],
                    "flags": report["flags"]
                    + (["alternative_effective_parent"] if candidate else []),
                },
                "candidate_nodes": candidate,
                "unique_scope": not candidate,
                "selection_uses_target_score": False,
            }
        )
    return results


def _span_text(data, span):
    """Recover addressed raw object text from retained sentence words."""
    for words in data["words"]:
        for w in words:
            if w["start"] <= span[0] < w["end"]:
                # Original sentence bytes are available on every predicate node.
                node = next(
                    n
                    for n in data["graph"]["nodes"]
                    if n["sentence_index"] == data["words"].index(words)
                )
                start = min(x["start"] for x in words)
                return node["projection_text"][span[0] - start : span[1] - start]
    raise ValueError("candidate span outside retained raw words")
