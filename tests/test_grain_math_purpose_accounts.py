#!/usr/bin/env python3
"""Synthetic mathematical and addressed-UD tests for HM-06."""

import numpy as np
import pytest

from grain_research.construction import math_purpose_accounts as P


def word(text, token, i, head, rel, upos="VERB", lemma=None):
    """Declare a valid raw-addressed word for a small synthetic dependency tree."""
    start = text.index(token)
    return {
        "id": i,
        "head": head,
        "deprel": rel,
        "upos": upos,
        "lemma": lemma or token.lower(),
        "start": start,
        "end": start + len(token),
        "surface": token,
    }


def record(rid="r", event="e", op="new", **extra):
    """Declare identity/version metadata independently of observed roles."""
    return {
        "record_id": rid,
        "event_id": event,
        "author_id": "u",
        "operation": op,
        "condition": -1,
        **extra,
    }


def test_same_actuality_cannot_determine_purpose():
    z = np.array([[1, 1, 1, -1], [1, -1, 1, -1]], float)
    mode = ["desire", "hypothetical"]
    np.testing.assert_equal(P.purpose_values(z, mode, "desire")[0], [1, 0])
    np.testing.assert_equal(P.purpose_values(z, mode, "hypothetical")[0], [0, -1])
    np.testing.assert_equal(
        P.purpose_values(z, mode, "desire", "universal_actual")[0], [0, 0]
    )
    np.testing.assert_equal(
        P.purpose_values(z, mode, "desire", "binary_mirror")[0], [0, 0]
    )


def test_state_exposure_is_not_action_or_emotion_polarity():
    z = np.array([[1, -1, 1, -1]], float)
    assert P.purpose_values(z, ["state"], "state")[0][0] == 1
    assert P.purpose_values(z, ["state"], "actual")[0][0] == 0


def test_version_replayed_before_qualification():
    a = record(mode="actual")
    b = record("b", op="revise", parent_id="r", mode="hypothetical")
    assert P.replay_versions([a, b]) == [b]
    assert P.finite_witnesses()["replay_before_route"]["actual"] == []


def test_alias_retract_and_revise_dependencies():
    a = record()
    alias = record("a", op="alias", parent_id="r")
    assert P.replay_versions([a, alias]) == [a]
    assert P.replay_versions([a, alias, record("b", op="retract", parent_id="r")]) == []
    with pytest.raises(ValueError, match="active"):
        P.replay_versions([a, record("b", op="revise", parent_id="absent")])
    with pytest.raises(ValueError, match="new event"):
        P.replay_versions([a, record("b")])
    with pytest.raises(ValueError, match="condition"):
        P.replay_versions([a, record("b", op="revise", parent_id="r", condition=1)])


def test_reference_population_and_no_fake_missing_zero():
    a = {"event_id": "a", "condition": -1, "value": 1}
    b = {"event_id": "b", "condition": 1, "value": -1}
    assert P.account_profile([a])["level"] is None
    assert P.account_profile([])["condition_choice_positive"] is None
    assert P.account_profile([a, b])["level"] == 0
    assert P.account_profile([a, b], 0.75)["level"] == -0.5
    with pytest.raises(ValueError, match="identities"):
        P.account_profile([a, a])


def test_open_complement_subject_and_mode_governor():
    text = "I want to organize volunteers."
    words = [
        word(text, "I", 1, 2, "nsubj", "PRON"),
        word(text, "want", 2, 0, "root"),
        word(text, "to", 3, 4, "mark", "PART"),
        word(text, "organize", 4, 2, "xcomp"),
        word(text, "volunteers", 5, 4, "obj", "NOUN"),
    ]
    observed = P.observe_modes([words], text)
    assert len(observed) == 1
    assert observed[0]["mode"] == "desire"
    assert observed[0]["subject_origin"] == "mode_governor"
    assert observed[0]["object"] == "volunteers"
    assert observed[0]["action_span"] == [10, 18]


def test_object_control_is_not_self_action():
    text = "I want Tom to organize volunteers."
    words = [
        word(text, "I", 1, 2, "nsubj", "PRON"),
        word(text, "want", 2, 0, "root"),
        word(text, "Tom", 3, 2, "obj", "PROPN"),
        word(text, "organize", 4, 2, "xcomp"),
        word(text, "volunteers", 5, 4, "obj", "NOUN"),
    ]
    e = P.observe_modes([words], text)[0]
    assert e["status"] == "unresolved"
    assert "unsupported_object_control" in e["flags"]


def test_source_and_input_validation():
    with pytest.raises(ValueError, match="nonfinite"):
        P.purpose_values(np.full((1, 4), np.nan), ["actual"], "actual")
    with pytest.raises(ValueError):
        P.purpose_values(np.ones((1, 4)), [], "actual")
    with pytest.raises(ValueError, match="one author"):
        P.replay_versions([record(), record("s", "f", author_id="v")])
