#!/usr/bin/env python3
"""Graph algebra tests; these fixtures do not test trained-parser accuracy."""

import pytest

from grain_research.source_event_graph import (
    event_graph,
    quote_spans,
    stanza_words,
    validate_words,
)


def graph(spec):
    """Build known synthetic dependencies with mechanically exact raw offsets."""
    text = " ".join(s[0] for s in spec)
    words, cursor = [], 0
    for i, (token, pos, head, dep, lemma) in enumerate(spec, 1):
        words.append(
            {
                "id": i,
                "text": token,
                "surface": token,
                "lemma": lemma,
                "upos": pos,
                "feats": "",
                "head": head,
                "deprel": dep,
                "start": cursor,
                "end": cursor + len(token),
            }
        )
        cursor += len(token) + 1
    return text, words, {e["predicate"]: e for e in event_graph(words, text)}


def test_direct_and_indirect_self_are_distinct_sources():
    _, _, events = graph(
        [
            ("She", "PRON", 2, "nsubj", "she"),
            ("said", "VERB", 0, "root", "say"),
            ("I", "PRON", 4, "nsubj", "i"),
            ("smile", "VERB", 2, "ccomp", "smile"),
        ]
    )
    assert events["smile"]["subject_reference"] == "writer_first_person_candidate"
    assert events["smile"]["source_candidate"] == "other_or_generic"


def test_quoted_first_person_shifts_without_deleting_words():
    text, words, events = graph(
        [
            ("She", "PRON", 2, "nsubj", "she"),
            ("said", "VERB", 0, "root", "say"),
            ('"', "PUNCT", 5, "punct", '"'),
            ("I", "PRON", 5, "nsubj", "i"),
            ("smile", "VERB", 2, "ccomp", "smile"),
            ('"', "PUNCT", 5, "punct", '"'),
        ]
    )
    assert events["smile"]["subject_reference"] == "reported_speaker_first_person"
    assert len(words) == 6 and '"' in text


def test_intention_is_not_actual_action():
    _, _, events = graph(
        [
            ("I", "PRON", 2, "nsubj", "i"),
            ("want", "VERB", 0, "root", "want"),
            ("to", "PART", 4, "mark", "to"),
            ("leave", "VERB", 2, "xcomp", "leave"),
        ]
    )
    assert events["leave"]["subject_origin"] == "inherited_candidate"
    assert events["leave"]["embedding_path"][0]["intention"]
    assert events["leave"]["factuality"] == "not_inferred"


def test_passive_reporting_recipient_not_source():
    _, _, events = graph(
        [
            ("I", "PRON", 3, "nsubj:pass", "i"),
            ("was", "AUX", 3, "aux:pass", "be"),
            ("told", "VERB", 0, "root", "tell"),
            ("I", "PRON", 6, "nsubj", "i"),
            ("was", "AUX", 6, "cop", "be"),
            ("friendly", "ADJ", 3, "ccomp", "friendly"),
        ]
    )
    assert events["friendly"]["source_candidate"] == "unresolved"
    assert (
        events["friendly"]["source_path"][0]["source_origin"]
        == "passive_source_unknown"
    )
    assert events["friendly"]["subject_reference"] == "writer_first_person_candidate"


def test_passive_by_agent_is_available_source():
    _, _, events = graph(
        [
            ("I", "PRON", 3, "nsubj:pass", "i"),
            ("was", "AUX", 3, "aux:pass", "be"),
            ("told", "VERB", 0, "root", "tell"),
            ("by", "ADP", 5, "case", "by"),
            ("Sam", "PROPN", 3, "obl:agent", "sam"),
            ("I", "PRON", 8, "nsubj", "i"),
            ("was", "AUX", 8, "cop", "be"),
            ("friendly", "ADJ", 3, "ccomp", "friendly"),
        ]
    )
    assert events["friendly"]["source_candidate"] == "other_or_generic"
    assert events["friendly"]["source_path"][0]["source_origin"] == "passive_agent"


def test_object_control_remains_ambiguous():
    _, _, events = graph(
        [
            ("I", "PRON", 2, "nsubj", "i"),
            ("told", "VERB", 0, "root", "tell"),
            ("him", "PRON", 2, "obj", "he"),
            ("to", "PART", 5, "mark", "to"),
            ("leave", "VERB", 2, "xcomp", "leave"),
        ]
    )
    assert events["leave"]["subject_origin"] == "ambiguous_controller"
    assert events["leave"]["participant_kind"] == "unresolved"


def test_negation_does_not_leak_to_embedded_clause():
    _, _, events = graph(
        [
            ("I", "PRON", 4, "nsubj", "i"),
            ("do", "AUX", 4, "aux", "do"),
            ("not", "PART", 4, "advmod", "not"),
            ("think", "VERB", 0, "root", "think"),
            ("she", "PRON", 6, "nsubj", "she"),
            ("left", "VERB", 4, "ccomp", "leave"),
        ]
    )
    assert events["think"]["local_negation"] == ["not"]
    assert events["leave"]["local_negation"] == []
    assert events["leave"]["embedding_path"][0]["local_negation"] == ["not"]


def test_condition_and_response_edges():
    _, _, events = graph(
        [
            ("If", "SCONJ", 3, "mark", "if"),
            ("I", "PRON", 3, "nsubj", "i"),
            ("worry", "VERB", 5, "advcl", "worry"),
            ("I", "PRON", 5, "nsubj", "i"),
            ("pause", "VERB", 0, "root", "pause"),
        ]
    )
    assert events["pause"]["adverbial_clauses"][0]["markers"] == ["if"]
    assert events["pause"]["adverbial_clauses"][0]["role"] == "governing_event"
    assert events["worry"]["adverbial_clauses"][0]["role"] == "subordinate_event"


def test_passive_subject_and_collective_are_not_agents():
    _, _, events = graph(
        [
            ("I", "PRON", 3, "nsubj:pass", "i"),
            ("was", "AUX", 3, "aux:pass", "be"),
            ("praised", "VERB", 0, "root", "praise"),
        ]
    )
    assert events["praise"]["passive_subject"]
    _, _, events = graph(
        [("We", "PRON", 2, "nsubj", "we"), ("run", "VERB", 0, "root", "run")]
    )
    assert events["run"]["participant_kind"] == "first_person_plural"


def test_subject_coordination_not_singular_author():
    _, _, events = graph(
        [
            ("I", "PRON", 4, "nsubj", "i"),
            ("and", "CCONJ", 3, "cc", "and"),
            ("Sam", "PROPN", 1, "conj", "sam"),
            ("run", "VERB", 0, "root", "run"),
        ]
    )
    assert events["run"]["participant_kind"] == "mixed"


def test_no_subject_stays_unknown():
    _, _, events = graph([("Run", "VERB", 0, "root", "run")])
    assert events["run"]["participant_kind"] == "unresolved"


def test_unpaired_quote_and_offset_failures():
    text, words, _ = graph(
        [
            ('"', "PUNCT", 3, "punct", '"'),
            ("I", "PRON", 3, "nsubj", "i"),
            ("run", "VERB", 0, "root", "run"),
        ]
    )
    assert not quote_spans(words)[0]["balanced"]
    broken = [dict(w) for w in words]
    broken[0]["head"] = 1
    with pytest.raises(ValueError, match="Cyclic"):
        validate_words(broken, text)
    words[0]["surface"] = "wrong"
    with pytest.raises(ValueError, match="source"):
        validate_words(words, text)


def test_expanded_words_retain_enclosing_raw_token():
    from types import SimpleNamespace as Obj

    word1 = Obj(
        id=1,
        text="can",
        lemma="can",
        upos="AUX",
        feats="",
        head=2,
        deprel="aux",
        start_char=None,
        end_char=None,
    )
    word2 = Obj(
        id=2,
        text="not",
        lemma="not",
        upos="PART",
        feats="",
        head=0,
        deprel="root",
        start_char=None,
        end_char=None,
    )
    token = Obj(start_char=0, end_char=5, words=[word1, word2])
    document = Obj(sentences=[Obj(tokens=[token], words=[word1, word2])])
    words = stanza_words(document, "can't")[0]
    assert all(w["expanded_multiword"] and w["surface"] == "can't" for w in words)
    assert words[0]["text"] != words[0]["surface"]
