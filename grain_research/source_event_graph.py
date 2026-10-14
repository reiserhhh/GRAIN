#!/usr/bin/env python3
"""Typed syntactic evidence graphs, preserving source and scope before scoring.

Neural dependency predictions are inputs. Graph rules are explicit reference
adapters, not a complete semantic-role/coreference/factuality model. No text is
deleted; unresolved roles and governing predicates stay attached to the event.
"""

from __future__ import annotations

from collections import defaultdict

REPORT = frozenset(
    [
        "say",
        "tell",
        "report",
        "claim",
        "write",
        "explain",
        "reply",
        "ask",
        "mention",
        "argue",
        "insist",
        "state",
        "announce",
        "describe",
        "add",
        "publish",
        "answer",
        "quote",
    ]
)
ATTITUDE = frozenset(
    [
        "think",
        "believe",
        "know",
        "remember",
        "learn",
        "discover",
        "forget",
        "admit",
        "suspect",
        "assume",
        "imagine",
        "realize",
        "feel",
        "doubt",
    ]
)
INTENTION = frozenset(
    [
        "want",
        "wish",
        "hope",
        "intend",
        "plan",
        "prefer",
        "try",
        "aim",
        "expect",
        "promise",
    ]
)
NEGATION = frozenset({"not", "n't", "never", "neither"})
FREQUENCY = frozenset(
    {"always", "usually", "often", "sometimes", "rarely", "seldom", "never"}
)
CLAUSE_EDGES = frozenset({"ccomp", "xcomp", "advcl", "acl", "conj", "parataxis"})


def validate_words(words, text):
    """Require a well-addressed acyclic dependency forest and known head IDs."""
    ids = {w["id"] for w in words}
    if len(ids) != len(words) or 0 in ids:
        raise ValueError("Invalid word IDs")
    by_id = {w["id"]: w for w in words}
    for w in words:
        if w["head"] != 0 and w["head"] not in ids:
            raise ValueError("Unknown dependency head")
        if not 0 <= w["start"] < w["end"] <= len(text):
            raise ValueError("Invalid raw offset")
        if text[w["start"] : w["end"]] != w["surface"]:
            raise ValueError("Raw source mismatch")
        seen, cur = {w["id"]}, w["head"]
        while cur:
            if cur in seen:
                raise ValueError("Cyclic dependency")
            seen.add(cur)
            cur = by_id[cur]["head"]


def stanza_words(document, text):
    """Export parser words; expanded contractions keep enclosing token offsets."""
    sentences = []
    for sentence_ix, sentence in enumerate(document.sentences):
        addresses = {}
        for token in sentence.tokens:
            for word in token.words:
                addresses[word.id] = (
                    token.start_char,
                    token.end_char,
                    len(token.words) > 1,
                )
        words = []
        for word in sentence.words:
            start, end, expanded = addresses[word.id]
            if word.start_char is not None and word.end_char is not None:
                start, end = word.start_char, word.end_char
            words.append(
                {
                    "id": word.id,
                    "sentence_index": sentence_ix,
                    "text": word.text,
                    "lemma": (word.lemma or word.text).lower(),
                    "upos": word.upos,
                    "feats": word.feats or "",
                    "head": word.head,
                    "deprel": word.deprel,
                    "start": int(start),
                    "end": int(end),
                    "surface": text[start:end],
                    "expanded_multiword": expanded,
                }
            )
        validate_words(words, text)
        sentences.append(words)
    return sentences


def quote_spans(words):
    """Pair parser punctuation, not arbitrary apostrophes inside words.

    Detached/unbalanced quotes are explicit unresolved spans. This detects
    orthography, not who spoke, irony or whether a quote was literally uttered.
    """
    pairs = {'"': '"', "“": "”", "‘": "’", "'": "'"}
    spans, stack = [], []
    seen = set()
    for w in sorted(words, key=lambda x: (x["start"], x["end"])):
        symbol = w["surface"]
        if w["upos"] != "PUNCT" or (w["start"], w["end"]) in seen:
            continue
        seen.add((w["start"], w["end"]))
        if stack and symbol == stack[-1]["closing"]:
            opening = stack.pop()
            spans.append({"start": opening["end"], "end": w["start"], "balanced": True})
        elif symbol in pairs:
            stack.append(
                {"start": w["start"], "end": w["end"], "closing": pairs[symbol]}
            )
        elif symbol in pairs.values():
            spans.append({"start": w["start"], "end": w["end"], "balanced": False})
    spans.extend(
        {"start": s["start"], "end": s["end"], "balanced": False} for s in stack
    )
    return spans


def participant_kind(subjects):
    """Syntactic participant class only; plural we is not singular author."""
    if not subjects:
        return "unresolved"
    lemmas = {s["lemma"] for s in subjects}
    if lemmas <= {"i", "me", "myself"}:
        return "first_person_singular"
    if lemmas <= {"we", "us", "ourselves"}:
        return "first_person_plural"
    if lemmas & {"i", "me", "we", "us"}:
        return "mixed"
    return "other_or_generic"


def event_graph(words, text, *, quote_context=None):
    """Convert a UD sentence into role-separated, original-addressed events."""
    validate_words(words, text)
    by_id = {w["id"]: w for w in words}
    children = defaultdict(list)
    for w in words:
        children[w["head"]].append(w)
    quotes = quote_spans(words) if quote_context is None else quote_context

    def edge(w):
        return w["deprel"].split(":")[0]

    def subjects(w):
        found = [s for s in children[w["id"]] if edge(s) in {"nsubj", "csubj"}]
        for subject in list(found):
            found.extend(c for c in children[subject["id"]] if edge(c) == "conj")
        return found

    def assertion_source(w):
        passive = (
            any(s["deprel"] == "nsubj:pass" for s in subjects(w))
            or "Voice=Pass" in w["feats"]
        )
        if passive:
            agents = [
                c
                for c in children[w["id"]]
                if c["deprel"] == "obl:agent"
                or (
                    edge(c) == "obl"
                    and any(
                        k["lemma"] == "by" and edge(k) == "case"
                        for k in children[c["id"]]
                    )
                )
            ]
            return agents, "passive_agent" if agents else "passive_source_unknown"
        return subjects(w), "active_subject_candidate"

    def ancestors(w):
        chain = []
        while w["head"]:
            w = by_id[w["head"]]
            chain.append(w)
        return chain

    def branch(w, *, stop_clauses=False):
        found = [w]
        for child in children[w["id"]]:
            if stop_clauses and edge(child) in CLAUSE_EDGES:
                continue
            found.extend(branch(child, stop_clauses=stop_clauses))
        return found

    predicates = [
        w
        for w in words
        if w["upos"] == "VERB"
        or (
            w["upos"] in {"ADJ", "NOUN", "PRON", "NUM", "ADV"}
            and (subjects(w) or any(edge(c) == "cop" for c in children[w["id"]]))
        )
    ]
    result = []
    for w in predicates:
        chain, subs = ancestors(w), subjects(w)
        subject_origin = "direct" if subs else "unresolved"
        if not subs and edge(w) in {"xcomp", "conj"} and chain:
            parent = chain[0]
            possible_object = [
                c for c in children[parent["id"]] if edge(c) in {"obj", "iobj"}
            ]
            if edge(w) == "xcomp" and possible_object:
                subject_origin = "ambiguous_controller"
            else:
                subs = subjects(parent)
                if subs:
                    subject_origin = "inherited_candidate"
        actor = participant_kind(subs)
        local = branch(w, stop_clauses=True)
        embeddings = []
        child = w
        for parent in chain:
            source_people, source_origin = assertion_source(parent)
            embeddings.append(
                {
                    "predicate_id": parent["id"],
                    "lemma": parent["lemma"],
                    "edge": edge(child),
                    "subject_ids": [s["id"] for s in subjects(parent)],
                    "subject_kind": participant_kind(subjects(parent)),
                    "source_ids": [s["id"] for s in source_people],
                    "source_kind": participant_kind(source_people),
                    "source_origin": source_origin,
                    "source_introducing": parent["lemma"] in REPORT | ATTITUDE,
                    "intention": parent["lemma"] in INTENTION,
                    "local_negation": [
                        c["lemma"]
                        for c in children[parent["id"]]
                        if c["lemma"] in NEGATION
                    ],
                }
            )
            child = parent
        source_paths = [
            e
            for e in embeddings
            if e["source_introducing"] and e["edge"] in {"ccomp", "xcomp", "parataxis"}
        ]
        source = (
            source_paths[0]["source_kind"]
            if source_paths
            else "document_writer_default"
        )
        in_quote = any(
            q["balanced"] and q["start"] <= w["start"] < q["end"] for q in quotes
        )
        reporting = [e for e in source_paths if e["lemma"] in REPORT]
        if in_quote and actor == "first_person_singular":
            if reporting and reporting[0]["source_kind"] == "other_or_generic":
                actor_reference = "reported_speaker_first_person"
            elif reporting and reporting[0]["source_kind"] == "first_person_singular":
                actor_reference = "writer_quoted_self_candidate"
            else:
                actor_reference = "quoted_first_person_unresolved"
        elif actor == "first_person_singular":
            actor_reference = "writer_first_person_candidate"
        else:
            actor_reference = actor
        conditions = []
        for clause in words:
            if edge(clause) != "advcl" or not (
                clause["head"] == w["id"] or clause["id"] == w["id"]
            ):
                continue
            marks = [c["lemma"] for c in children[clause["id"]] if edge(c) == "mark"]
            content = branch(clause)
            conditions.append(
                {
                    "head_id": clause["id"],
                    "governs": clause["head"],
                    "markers": marks,
                    "start": min(c["start"] for c in content),
                    "end": max(c["end"] for c in content),
                    "role": "subordinate_event"
                    if clause["id"] == w["id"]
                    else "governing_event",
                }
            )
        local_ids = {c["id"] for c in local}
        result.append(
            {
                "head_id": w["id"],
                "sentence_index": w.get("sentence_index", 0),
                "predicate": w["lemma"],
                "predicate_start": w["start"],
                "predicate_end": w["end"],
                "subject_ids": [s["id"] for s in subs],
                "subject_origin": subject_origin,
                "participant_kind": actor,
                "subject_reference": actor_reference,
                "source_candidate": source,
                "source_path": source_paths,
                "embedding_path": embeddings,
                "local_word_ids": sorted(local_ids),
                "span_start": min(c["start"] for c in local),
                "span_end": max(c["end"] for c in local),
                "passive_subject": any(s["deprel"] == "nsubj:pass" for s in subs),
                "local_negation": [c["lemma"] for c in local if c["lemma"] in NEGATION],
                "modal_cues": [
                    c["lemma"]
                    for c in children[w["id"]]
                    if edge(c) == "aux"
                    and c["lemma"]
                    in {
                        "can",
                        "could",
                        "may",
                        "might",
                        "must",
                        "should",
                        "would",
                        "will",
                        "shall",
                    }
                ],
                "frequency_cues": [
                    c["lemma"] for c in local if c["lemma"] in FREQUENCY
                ],
                "morphology": w["feats"],
                "adverbial_clauses": conditions,
                "inside_quote": in_quote,
                "unbalanced_quote_present": any(not q["balanced"] for q in quotes),
                "factuality": "not_inferred",
                "query_relevance": "not_inferred",
            }
        )
    return result
