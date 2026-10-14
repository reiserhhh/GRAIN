#!/usr/bin/env python3
"""Extraction, scope and deterministic localization tests with known toy graphs."""

import numpy as np
import pytest
from test_grain_source_event_graph import graph

from grain_research.query_event import (
    event_views,
    evidence_route,
    rank_events,
    render_fragments,
)


def test_negated_ancestor_survives_scope_not_local():
    text, words, events = graph(
        [
            ("I", "PRON", 4, "nsubj", "i"),
            ("do", "AUX", 4, "aux", "do"),
            ("not", "PART", 4, "advmod", "not"),
            ("think", "VERB", 0, "root", "think"),
            ("she", "PRON", 6, "nsubj", "she"),
            ("left", "VERB", 4, "ccomp", "leave"),
        ]
    )
    v = event_views(text, words, events["leave"])
    assert v["local_text"] == "she left"
    assert v["scope_text"] == text
    assert evidence_route(events["leave"])["ancestor_negation"]


def test_inherited_subject_is_not_rewritten_as_observed_action():
    text, words, events = graph(
        [
            ("I", "PRON", 2, "nsubj", "i"),
            ("want", "VERB", 0, "root", "want"),
            ("to", "PART", 4, "mark", "to"),
            ("help", "VERB", 2, "xcomp", "help"),
        ]
    )
    v = event_views(text, words, events["help"])
    assert v["local_text"] == "I ... to help"
    assert v["scope_text"] == "I want to help"
    assert evidence_route(events["help"])["intention_embedded"]


def test_condition_is_not_discarded_from_response():
    text, words, events = graph(
        [
            ("If", "SCONJ", 3, "mark", "if"),
            ("I", "PRON", 3, "nsubj", "i"),
            ("worry", "VERB", 5, "advcl", "worry"),
            ("I", "PRON", 5, "nsubj", "i"),
            ("pause", "VERB", 0, "root", "pause"),
        ]
    )
    v = event_views(text, words, events["pause"])
    assert v["local_text"] == "I pause"
    assert v["scope_text"] == text


def test_surface_span_union_does_not_duplicate_contractions():
    words = [{"id": 1, "start": 0, "end": 5}, {"id": 2, "start": 0, "end": 5}]
    rendered, spans = render_fragments("can't", words, {1, 2})
    assert rendered == "can't" and spans == [[0, 5]]


def test_role_not_forced_to_writer_in_passive_report():
    _, _, events = graph(
        [
            ("I", "PRON", 3, "nsubj:pass", "i"),
            ("was", "AUX", 3, "aux:pass", "be"),
            ("told", "VERB", 0, "root", "tell"),
            ("I", "PRON", 5, "nsubj", "i"),
            ("help", "VERB", 3, "ccomp", "help"),
        ]
    )
    assert evidence_route(events["help"])["route"] == "writer_subject_embedded_source"
    assert evidence_route(events["help"])["source_candidate"] == "unresolved"


def test_exact_score_ties_have_stable_not_incidental_winner():
    r = rank_events(["b", "a", "c"], [0.4, 0.4, 0.2])
    assert r["index"] == 1 and r["near_ties"] == 2 and r["margin"] == 0


@pytest.mark.parametrize(
    "ids,scores", [([], []), (["a", "a"], [0.1, 0.2]), (["a"], [np.nan])]
)
def test_invalid_rank_input_fails(ids, scores):
    with pytest.raises(ValueError):
        rank_events(ids, scores)
