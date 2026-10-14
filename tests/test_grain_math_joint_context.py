#!/usr/bin/env python3
"""Cold-material, addressed-view and finite conditional-function tests."""

import numpy as np
import pytest

from grain_research.construction import math_joint_context as J
from grain_research.construction import math_observation_dag as D
from tests.test_grain_math_purpose_accounts import word


def graph_words(text, explicit=False, comma=False):
    """Construct a minimal UD fixture, not a replacement for the actual parser."""
    words = [
        word(text, "I", 1, 2, "nsubj", "PRON"),
        word(text, "organize", 2, 0, "root"),
        word(text, "teams", 3, 2, "obj", "NOUN"),
        word(text, "preparing", 4, 2, "advcl", lemma="prepare"),
        word(text, "concerts", 5, 4, "obj", "NOUN"),
    ]
    if explicit:
        words += [
            word(text, "Tom", 6, 4, "nsubj", "PROPN"),
            word(text, "while", 7, 4, "mark", "SCONJ"),
        ]
    if comma:
        words += [word(text, ",", 6, 4, "punct", "PUNCT")]
    return words


def test_candidate_retains_context_without_new_event():
    text = "I organize teams preparing concerts."
    words = graph_words(text)
    graph = D.observe_dag([words], text)
    e = J.object_views([words], text, graph)[0]
    assert e["object"] == "teams"
    assert e["legal_object"] == "teams preparing concerts"
    assert len(e["candidates"]) == 2 and len(graph["events"]) == 1
    assert e["contexts"][0]["actor"] == "unknown"
    assert not e["contexts"][0]["event_identity_inherited"]


def test_explicit_actor_is_context_not_an_object_extension():
    text = "I organize teams while Tom preparing concerts."
    words = graph_words(text, explicit=True)
    e = J.object_views([words], text, D.observe_dag([words], text))[0]
    assert e["legal_object"] == "teams"
    assert "Tom" in e["naive_object"]
    assert len(e["candidates"]) == 1


def test_comma_blocks_attachment_but_not_context_retention():
    text = "I organize teams, preparing concerts."
    words = graph_words(text, comma=True)
    e = J.object_views([words], text, D.observe_dag([words], text))[0]
    assert e["legal_object"] == "teams"
    assert e["contexts"][0]["comma_boundary"]
    assert "preparing" in e["naive_object"]


def test_envelope_permutation_invariance_and_expansion():
    a = J.envelope([-0.2, 0.3])
    assert a == J.envelope([0.3, -0.2])
    assert not a["all_positive"] and a["any_positive"] and a["gate_disagreement"]
    b = J.envelope([-0.7, -0.2, 0.3, 0.4])
    assert b["low"] <= a["low"] <= a["high"] <= b["high"]
    with pytest.raises(ValueError):
        J.envelope([])
    with pytest.raises(ValueError):
        J.envelope([float("nan")])


def test_four_corner_annihilates_additive_not_interaction():
    assert J.cross_difference(2, 5, 4, 7) == 0
    assert J.cross_difference(0, 0, 0, 1) == 1
    x = np.arange(12).reshape(3, 4)
    assert np.all(J.cross_difference(x, x + 1, x + 2, x + 5) == 2)


def test_target_labels_cannot_change_observed_views():
    text = "I organize teams preparing concerts."
    words = graph_words(text)
    a = D.observe_dag([words], text)
    b = D.observe_dag([words], text)
    b["unrelated_expected_target"] = "a different hidden design label"
    assert J.object_views([words], text, a) == J.object_views([words], text, b)


def test_bank_count_balance_and_fresh_public_texts():
    source = [
        {
            "target": t,
            "action_index": i,
            "object_index": 0,
            "bits": [1, 1, 1, 1],
            "action": f"act{i}",
        }
        for t in ("coordination_action", "verification_action")
        for i in range(8)
    ]
    bank = J.material_bank(source)
    assert len(bank) == 2656
    assert len({r["material_id"] for r in bank}) == len(bank)
    assert len({r["text"] for r in bank}) == len(bank)
    assert {r["events"][0]["mode"] for r in bank} == set(J.P.ACCOUNTS)
    for r in bank:
        e = r["events"][0]
        assert r["text"][slice(*e["action_span"])] == e["action"]
        assert e["object"] in r["text"]


def test_typed_function_does_not_factor_blind_sum():
    witness = J.kernel_factorization_witness()
    assert witness["collapsed"] == 0 and witness["qualified"] == 1
