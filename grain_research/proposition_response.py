#!/usr/bin/env python3
"""Dependency-addressed propositions and limited reply binding for measurement.

This is a bounded adapter of learned UD parses, not a factuality or speech-act
oracle. Conditions are extracted phrases, not named psychological situations.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from grain_research.source_event_graph import event_graph, quote_spans

INTENT = {"intend", "plan", "want", "hope", "aim", "wish"}
PREFERENCE = {"prefer", "like", "enjoy"}
ABILITY = {"able", "capable"}
STATE = {"feel", "seem"}
REQUEST = {"ask", "request", "seek"}
SUPPORT = {"help", "assistance", "support", "reassurance", "confirmation"}


def digest_text(text):
    """Use exact original Unicode text for provenance."""
    return hashlib.sha256(text.encode()).hexdigest()


def frames(text, sentences, speaker):
    """Expose semantic candidates using raw parse edges, never stimulus labels."""
    quote_context = quote_spans([w for sentence in sentences for w in sentence])
    result = []
    for words in sentences:
        indexed = {w["id"]: w for w in words}
        events = event_graph(words, text, quote_context=quote_context)
        sentence_text = text[
            min(w["start"] for w in words) : max(w["end"] for w in words)
        ]
        question = sentence_text.rstrip().endswith("?")
        for event in events:
            head = indexed[event["head_id"]]
            chain = event["embedding_path"]
            local = [indexed[i] for i in event["local_word_ids"]]
            lemmas = {w["lemma"] for w in local}
            actor_lemmas = {indexed[i]["lemma"] for i in event["subject_ids"]}
            actor = "unresolved"
            if actor_lemmas and actor_lemmas <= {"i", "me", "myself"}:
                actor = speaker
            elif actor_lemmas == {"you"}:
                actor = "respondent" if speaker == "partner" else "partner"
            elif actor_lemmas:
                actor = "other"
            unsafe_source = event["inside_quote"] or bool(event["source_path"])
            if unsafe_source:
                actor = "source_scoped_unresolved"
            governing = {x["lemma"] for x in chain} | {event["predicate"]}
            if governing & INTENT:
                channel = "intention"
            elif governing & PREFERENCE:
                channel = "preference"
            elif governing & ABILITY or set(event["modal_cues"]) & {"can", "could"}:
                channel = "ability_candidate"
            elif governing & STATE:
                channel = "current_experience_candidate"
            elif "will" in event["modal_cues"]:
                channel = "prospective"
            elif "Tense=Past" in head["feats"] or any(
                "Tense=Past" in w["feats"]
                and w["deprel"].split(":")[0] == "aux"
                and w["head"] == head["id"]
                for w in words
            ):
                channel = "past_action_report"
            else:
                channel = "unspecified_expression"
            if (
                not question
                and speaker == "respondent"
                and not actor_lemmas
                and not chain
                and ("please" in lemmas or "Mood=Imp" in head["feats"])
            ):
                channel = "current_request_candidate"
            conditions = []
            for owner in [event] + [
                e for e in events if e["head_id"] in {x["predicate_id"] for x in chain}
            ]:
                for condition in owner["adverbial_clauses"]:
                    if (
                        condition["role"] == "governing_event"
                        and condition not in conditions
                    ):
                        conditions.append(condition)
            addresses = [{**c, "text": text[c["start"] : c["end"]]} for c in conditions]
            negative_scopes = int(bool(event["local_negation"])) + sum(
                bool(x["local_negation"])
                for x in chain
                if x["lemma"] in INTENT | PREFERENCE | ABILITY
            )
            result.append(
                {
                    "id": f"{speaker}:{event['sentence_index']}:{event['head_id']}",
                    "source_sha256": digest_text(text),
                    "speaker": speaker,
                    "predicate": event["predicate"],
                    "start": event["predicate_start"],
                    "end": event["predicate_end"],
                    "quote": text[event["predicate_start"] : event["predicate_end"]],
                    "actor": actor,
                    "channel": channel,
                    "question": question,
                    "sign": -1 if negative_scopes == 1 else 1,
                    "ambiguous_negation": negative_scopes > 1,
                    "source_scoped": unsafe_source,
                    "conditions": addresses,
                    "target_relevant": event["predicate"] in REQUEST
                    and bool(lemmas & SUPPORT),
                    "local_negation": event["local_negation"],
                    "modal_cues": event["modal_cues"],
                    "embedding_path": chain,
                    "factuality": "not_observed",
                }
            )
    return result


def bind_reply(partner, reply, partner_frames, reply_frames):
    """Bind a bare answer only to a unique explicit positive target proposition.

    A negative interrogative is ambiguous across response conventions. Explicit
    respondent propositions stand on their own; matching words are only a
    dependency candidate, not an asserted correction or coreference decision.
    """
    targets = [f for f in partner_frames if f["target_relevant"] and f["question"]]
    own = [f for f in reply_frames if f["target_relevant"] and not f["question"]]
    answer = re.fullmatch(r"\s*(yes|no)[.!]?\s*", reply, re.IGNORECASE)
    base = {
        "partner_sha256": digest_text(partner),
        "reply_sha256": digest_text(reply),
        "antecedents": [f["id"] for f in targets],
        "observations": [],
        "reply_span": [0, len(reply)],
        "status": "unresolved",
    }
    if own:
        base.update(
            status="explicit_reply_proposition",
            observations=[{**f, "evidence_kind": "respondent_statement"} for f in own],
        )
    elif answer and len(targets) == 1:
        frame = targets[0]
        if frame["sign"] < 0 or frame["ambiguous_negation"] or frame["source_scoped"]:
            base["status"] = "ambiguous_question_scope"
        else:
            base.update(
                status="bound_bare_answer",
                observations=[
                    {
                        **frame,
                        "sign": 1 if answer.group(1).lower() == "yes" else -1,
                        "evidence_kind": "answer_to_proposition",
                        "antecedent_id": frame["id"],
                        "answer_source_sha256": digest_text(reply),
                        "answer_span": [0, len(reply)],
                    }
                ],
            )
    elif answer:
        base["status"] = "no_unique_target_proposition"
    return base


def response_coordinates(binding):
    """Signed self-report/intent evidence, with unsupported observations missing."""
    observations = binding["observations"]
    result = {"past_action_report": 0.0, "intention": 0.0}
    eligible = [
        x
        for x in observations
        if x["channel"] in result
        and not x["ambiguous_negation"]
        and not x["source_scoped"]
        and x["actor"] != "unresolved"
    ]
    if not eligible:
        return {k: None for k in result}, False
    for row in eligible:
        if row["actor"] == "respondent":
            result[row["channel"]] += row["sign"]
    return result, True


def condition_text(binding):
    """Return extracted condition text or missing, without a global taxonomy."""
    values = []
    for frame in binding["observations"]:
        for condition in frame["conditions"]:
            if condition["text"] not in values:
                values.append(condition["text"])
    return " ; ".join(values)


def response_profile(records, conditions):
    """Keep observation coverage apart from the observed conditional mean."""
    values = {}
    for condition in conditions:
        group = [r for r in records if r["condition"] == condition]
        accepted = [
            r for r in group if r["score"] is not None and np.isfinite(r["score"])
        ]
        values[str(condition)] = {
            "opportunities": len(group),
            "observed": len(accepted),
            "score": float(np.mean([r["score"] for r in accepted]))
            if accepted
            else None,
        }
    return values
