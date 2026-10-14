#!/usr/bin/env python3
"""Synthetic evidence tests require no corpus, weights, or network access."""

import numpy as np
import pytest

from grain_research.construction import math_evidence as E
from grain_research.evidence_history import fit_source, kernels


def record(rid, event, c, value=1.0, **extras):
    return {
        "record_id": rid,
        "event_id": event,
        "condition": c,
        "author_id": "u1",
        "block_id": "b0",
        "value": value,
        "kind": "new",
        "qualification": "own_actual",
        **extras,
    }


def test_slice_preserves_original_offsets():
    text = "We needed a plan. I arranged the group meeting yesterday. Others agreed."
    s = E.candidate_slice(text)
    assert s["fragment"] == text[s["start"] : s["end"]]
    assert s["condition"] is None
    assert (
        E.candidate_slice('Someone said "I arranged the group meeting."')["status"]
        == "quotation_candidate"
    )


def test_denial_is_not_labeled_as_waiting():
    variants = E.probe_variants("I arranged the group meeting", "arranged")
    assert "did not arrange" in variants["denied_action"]
    assert "waited" not in variants["denied_action"]
    assert "would arrange" in variants["hypothetical"]


def test_unknown_and_other_remain_missing():
    state = E.replay_qualified(
        [
            record("r1", "e1", None),
            record("r2", "e2", 1, qualification="other_actual"),
        ]
    )
    assert not state["records"]
    assert E.fixed_reference_profile(state["records"])["level"] is None
    assert len(state["excluded"]) == 2


def test_reference_does_not_change_with_choice_frequency():
    baseline = [record("r1", "e1", -1, -1), record("r2", "e2", 1, 1)]
    more = baseline + [record(f"r{i}", f"e{i}", 1, 1) for i in range(3, 10)]
    a, b = E.fixed_reference_profile(baseline), E.fixed_reference_profile(more)
    assert a["level"] == b["level"] == 0
    assert a["response"] == b["response"] == 1
    assert a["condition_choice_positive"] != b["condition_choice_positive"]


def test_missing_condition_not_renormalized():
    profile = E.fixed_reference_profile([record("r", "e", 1)])
    assert profile["endpoints"]["1"] == 1
    assert profile["level"] is None and profile["response"] is None


def test_alias_revision_and_withdrawal():
    history = [record("r1", "e1", -1), record("r2", "e2", 1)]
    alias = {"record_id": "a", "kind": "alias", "parent_id": "r1", "author_id": "u1"}
    a = E.replay_qualified(history)
    b = E.replay_qualified(history + [alias])
    assert E.fixed_reference_profile(a["records"]) == E.fixed_reference_profile(
        b["records"]
    )
    revision = record("v", "e1", -1, -1, kind="revise", parent_id="r1")
    revised = E.replay_qualified(history + [alias, revision])
    assert revised["active_events"] == 2
    assert "a" in revised["stale_derived"]
    gone = E.replay_qualified(
        history
        + [
            revision,
            {"record_id": "x", "kind": "retract", "parent_id": "v", "author_id": "u1"},
        ]
    )
    assert gone["active_events"] == 1


def test_author_and_event_scope_are_not_guessed():
    with pytest.raises(ValueError, match="One author"):
        E.replay_qualified(
            [record("a", "e1", -1), record("b", "e2", 1, author_id="u2")]
        )


def test_revision_can_remove_old_qualification():
    old = record("old", "event", 1, 0.8)
    correction = record(
        "corrected",
        "event",
        1,
        None,
        kind="revise",
        parent_id="old",
        qualification="other_actual",
    )
    state = E.replay_qualified([old, correction])
    assert not state["records"]
    assert state["excluded"][0]["record_id"] == "corrected"
    assert E.fixed_reference_profile(state["records"])["level"] is None


def test_revision_can_add_previously_missing_qualification():
    old = record("old", "event", None, None, qualification="unknown")
    correction = record("corrected", "event", 1, 0.8, kind="revise", parent_id="old")
    state = E.replay_qualified([old, correction])
    assert state["active_events"] == 1
    assert state["records"][0]["record_id"] == "corrected"
    with pytest.raises(ValueError, match="conflicting context"):
        E.replay_qualified([record("a", "e1", -1), record("b", "e1", 1, kind="repeat")])


def test_schedule_components_identifiable():
    schedule = E.panel_schedule()
    _, bases = kernels(schedule)
    assert np.linalg.matrix_rank(bases.reshape(5, -1).T) == 5
    assert len(schedule) == 32
    assert len({r["event_id"] for r in schedule}) == 16


def test_panels_fit_without_access_to_truth():
    lookup = {
        (f, p, r, c): float(p)
        for f in range(4)
        for p in (-1, 1)
        for r in (-1, 1)
        for c in (-1, 1)
    }
    observed, truth = E.simulate_panels(lookup, [0, 1], 32, 7)
    assert set(observed) == {"values", "records"}
    fitted = fit_source(**observed)
    assert fitted["source_authors"] == 32
    assert truth["level_response"].shape == (32, 2)


def test_invalid_reference_and_score_rejected():
    with pytest.raises(ValueError):
        E.fixed_reference_profile([], 1.1)
    with pytest.raises(ValueError, match="finite"):
        E.replay_qualified([record("r", "e", 1, np.nan)])
