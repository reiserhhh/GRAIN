#!/usr/bin/env python3
"""Role-binding tests use addressed public syntax and mathematical witnesses."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_binding as B
from grain_research.construction import math_local_roles as R


def words(text, dependencies):
    """Make exact raw-addressed tokens for controlled dependency unit tests."""
    tokens = list(re.finditer(r"\w+|[,.;]", text))
    assert len(tokens) == len(dependencies)
    return [
        {
            "id": i + 1,
            "head": head,
            "deprel": edge,
            "upos": upos,
            "lemma": m.group().lower(),
            "text": m.group(),
            "surface": m.group(),
            "feats": "VerbForm=Inf" if upos == "VERB" else "",
            "start": m.start(),
            "end": m.end(),
        }
        for i, (m, (head, edge, upos)) in enumerate(
            zip(tokens, dependencies, strict=True)
        )
    ]


def bank():
    spec = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/grain_math_binding_v1.json"
        ).read_text()
    )
    return B.material_bank(spec)


def test_role_design_has_full_cells_and_no_source_objects():
    source = bank()
    rows = R.material_bank(source)
    assert len(rows) == 2368
    main = [r for r in rows if r["kind"] == "mixture"]
    assert len(main) == 2048
    assert len({tuple(r["roles"]) for r in main}) == 64
    assert all(
        source[e["source_index"]]["role"] == "unseen_object"
        for r in rows
        for e in r["events"]
    )
    assert sum(r["ambiguous"] for r in rows) == 16
    assert all(
        r["text"][e["span"][0] : e["span"][1]] for r in rows for e in r["events"]
    )


def test_explicit_foreign_subject_and_negation_do_not_leak():
    text = "I did not organize volunteers but my colleague did coordinate workers."
    w = words(
        text,
        [
            (4, "nsubj", "PRON"),
            (4, "aux", "AUX"),
            (4, "advmod", "PART"),
            (0, "root", "VERB"),
            (4, "obj", "NOUN"),
            (10, "cc", "CCONJ"),
            (8, "nmod:poss", "PRON"),
            (10, "nsubj", "NOUN"),
            (10, "aux", "AUX"),
            (4, "conj", "VERB"),
            (10, "obj", "NOUN"),
            (4, "punct", "PUNCT"),
        ],
    )
    events = R.project_events([w], text)
    assert len(events) == 2
    assert events[0]["projection_text"] == "I did not organize volunteers."
    assert events[1]["projection_text"] == "my colleague did coordinate workers."
    assert all(e["status"] == "ok" for e in events)
    assert all(e["subject_origin"] == "direct" for e in events)
    assert "not" not in events[1]["projection_text"]


def test_missing_shared_subject_and_auxiliary_have_addresses():
    text = "I did organize volunteers and coordinate workers."
    w = words(
        text,
        [
            (3, "nsubj", "PRON"),
            (3, "aux", "AUX"),
            (0, "root", "VERB"),
            (3, "obj", "NOUN"),
            (6, "cc", "CCONJ"),
            (3, "conj", "VERB"),
            (6, "obj", "NOUN"),
            (3, "punct", "PUNCT"),
        ],
    )
    second = R.project_events([w], text)[1]
    assert second["projection_text"] == "I did coordinate workers."
    assert second["subject_origin"] == second["auxiliary_origin"] == "coordination_head"
    assert second["inherited_word_ids"]["subject"] == [1]
    assert second["status"] == "ok"
    assert all(
        text[a:b] == p["surface"] for p in second["pieces"] for a, b in [p["span"]]
    )


def test_shared_negation_remains_unresolved():
    text = "I did not organize volunteers and coordinate workers."
    w = words(
        text,
        [
            (4, "nsubj", "PRON"),
            (4, "aux", "AUX"),
            (4, "advmod", "PART"),
            (0, "root", "VERB"),
            (4, "obj", "NOUN"),
            (7, "cc", "CCONJ"),
            (4, "conj", "VERB"),
            (7, "obj", "NOUN"),
            (4, "punct", "PUNCT"),
        ],
    )
    second = R.project_events([w], text)[1]
    assert second["status"] == "unresolved"
    assert "ambiguous_shared_negation" in second["flags"]


def test_identity_binding_uses_address_not_role_truth():
    projected = [{"action_span": [10, 14], "flags": [], "status": "ok"}]
    supplied = [{"event_id": "a", "span": [0, 20], "bits": [-1, -1, -1, -1]}]
    bound = R.attach_event_identity(projected, supplied)
    assert bound[0]["event_id"] == "a"
    assert "bits" not in bound[0]
    supplied[0]["bits"] = [1, 1, 1, 1]
    assert R.attach_event_identity(projected, supplied) == bound


def test_alias_does_not_add_observation_and_new_event_can_update():
    ids = [{"event_id": "a"}, {"event_id": "b"}]
    base = R.event_profile(ids, np.array([-1, 0.0]), np.array([True, False]))
    alias = R.event_profile(
        ids + [ids[0]], np.array([-1, 0.0, -0.9]), np.array([True, False, True])
    )
    new = R.event_profile(
        ids + [{"event_id": "c"}],
        np.array([-1, 0.0, 1.0]),
        np.array([True, False, True]),
    )
    assert base["score"] == alias["score"] == -1
    assert alias["unique_events"] == 2
    assert alias["alias_score_variation"] == pytest.approx(0.1)
    assert new["score"] == 0 and new["observed_events"] == 2


def test_equal_event_count_does_not_hide_uncovered_records():
    duplicate = [
        {"status": "ok", "record_index": 0},
        {"status": "ok", "record_index": 0},
    ]
    assert R.missing_records(duplicate, 2) == 2
    assert (
        R.missing_records(
            [{"status": "ok", "record_index": 0}, {"status": "ok", "record_index": 1}],
            2,
        )
        == 0
    )


def test_absence_and_incomplete_evidence_are_not_neutral_authors():
    absent = R.event_profile([{"event_id": "a"}], np.array([0.0]), np.array([False]))
    incomplete = R.event_profile(
        [{"event_id": "a"}], np.array([-1.0]), np.array([True]), unresolved=1
    )
    assert absent["score"] is None and absent["complete"]
    assert incomplete["score"] is None and incomplete["known_score"] == -1
    assert not incomplete["complete"] and incomplete["unit"] is None


def test_equal_marginals_are_not_a_sufficient_role_statistic():
    witness = R.marginal_collision_witness()
    assert witness["marginals_equal"]
    assert witness["left_author_expression"] == -1
    assert witness["right_author_expression"] == 1
    assert witness["joint_po_moment"] == [-1, 1]
