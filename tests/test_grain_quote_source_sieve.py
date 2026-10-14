#!/usr/bin/env python3
"""Reporting-source algebra fixtures, not trained-parser accuracy tests."""

import pytest

from grain_research.quote_source_sieve import explicit_quote_source


def fixture(subject="I", negated=False, modal=False, dependency=True):
    """Construct exact UD addresses for one attributed quotation."""
    spec = [
        (subject, "PRON", 2, "nsubj", subject.lower()),
        ("said", "VERB", 0, "root", "say"),
        ('"', "PUNCT", 5, "punct", '"'),
        ("I", "PRON", 5, "nsubj", "i"),
        (
            "smile",
            "VERB",
            2 if dependency else 0,
            "ccomp" if dependency else "root",
            "smile",
        ),
        ('"', "PUNCT", 5, "punct", '"'),
    ]
    if negated:
        spec.append(("not", "PART", 2, "advmod", "not"))
    if modal:
        spec.append(("might", "AUX", 2, "aux", "might"))
    text = " ".join(s[0] for s in spec)
    words, cursor = [], 0
    for i, (token, upos, head, deprel, lemma) in enumerate(spec, 1):
        words.append(
            {
                "id": i,
                "text": token,
                "surface": token,
                "upos": upos,
                "head": head,
                "deprel": deprel,
                "lemma": lemma,
                "feats": "",
                "start": cursor,
                "end": cursor + len(token),
            }
        )
        cursor += len(token) + 1
    return text, words, words[3]["start"], words[4]["end"]


FALLBACK = {"source_owner": "unresolved", "reason": "no_entailment"}


@pytest.mark.parametrize("subject,owner", [("I", "self"), ("She", "other")])
def test_explicit_subject_not_quoted_first_person(subject, owner):
    text, words, a, b = fixture(subject)
    result = explicit_quote_source(text, [words], a, b, FALLBACK)
    assert result["source_owner"] == owner
    assert result["mechanism"] == "dependency_report_source"
    assert text[slice(*result["cue_span"])] == "said"
    assert [text[slice(*s)] for s in result["source_spans"]] == [subject]


def test_preceding_same_sentence_cue_is_distinct_heuristic():
    text, words, a, b = fixture("She", dependency=False)
    result = explicit_quote_source(text, [words], a, b, FALLBACK)
    assert result["source_owner"] == "other"
    assert result["mechanism"] == "same_sentence_preceding_report_source"


@pytest.mark.parametrize("kwargs", [{"negated": True}, {"modal": True}])
def test_uncertain_reporting_does_not_use_fallback_to_force_self(kwargs):
    text, words, a, b = fixture(**kwargs)
    result = explicit_quote_source(
        text, [words], a, b, {"source_owner": "self", "reason": "nli"}
    )
    assert result["source_owner"] == "unresolved"
    assert result["mechanism"] == "dependency_report_source"


def test_no_report_cue_preserves_nli_fallback():
    text, words, a, b = fixture()
    words[1]["lemma"] = "think"
    result = explicit_quote_source(text, [words], a, b, FALLBACK)
    assert result["source_owner"] == "unresolved"
    assert result["mechanism"] == "nli_fallback"


def test_no_arbitrary_cross_sentence_source():
    text, words, a, b = fixture("She")
    source, quote = words[:2], words[2:]
    quote = [
        dict(w, id=w["id"] - 2, head=w["head"] - 2 if w["head"] > 2 else 0)
        for w in quote
    ]
    result = explicit_quote_source(text, [source, quote], a, b, FALLBACK)
    assert result["mechanism"] == "nli_fallback"


def test_bad_address_rejected():
    text, words, _, _ = fixture()
    with pytest.raises(ValueError, match="quote address"):
        explicit_quote_source(text, [words], -1, len(text), FALLBACK)
