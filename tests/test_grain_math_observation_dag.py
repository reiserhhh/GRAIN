#!/usr/bin/env python3
"""Finite algebra and synthetic addressed-syntax checks for HM-07."""

import numpy as np
import pytest

from grain_research.construction import math_observation_dag as D
from tests.test_grain_math_purpose_accounts import word


def test_context_not_duplicate_outer_event():
    text = "I want to organize teams preparing concerts."
    words = [
        word(text, "I", 1, 2, "nsubj", "PRON"),
        word(text, "want", 2, 0, "root"),
        word(text, "organize", 3, 2, "xcomp"),
        word(text, "teams", 4, 3, "obj", "NOUN"),
        word(text, "preparing", 5, 3, "advcl", lemma="prepare"),
        word(text, "concerts", 6, 5, "obj", "NOUN"),
    ]
    graph = D.observe_dag([words], text)
    assert len(graph["events"]) == 1
    assert len(graph["nodes"]) == 3
    assert graph["nodes"][2]["type"] == "adjunct_context"
    assert graph["nodes"][2]["actor"] == "unknown"
    assert not graph["nodes"][2]["root_event_identity_inherited"]
    assert graph["events"][0]["mode"] == "desire"


def test_object_context_keeps_own_actor():
    text = "I organize teams that Tom trains."
    words = [
        word(text, "I", 1, 2, "nsubj", "PRON"),
        word(text, "organize", 2, 0, "root"),
        word(text, "teams", 3, 2, "obj", "NOUN"),
        word(text, "that", 4, 6, "obj", "PRON"),
        word(text, "Tom", 5, 6, "nsubj", "PROPN"),
        word(text, "trains", 6, 3, "acl:relcl", lemma="train"),
    ]
    graph = D.observe_dag([words], text)
    assert len(graph["events"]) == 1
    assert graph["nodes"][1]["type"] == "object_context"
    assert graph["nodes"][1]["actor"] == "other"
    assert "Tom trains" in graph["events"][0]["object"]


def test_conceiver_and_actor_are_not_same_role():
    text = "I want Tom to organize teams."
    words = [
        word(text, "I", 1, 2, "nsubj", "PRON"),
        word(text, "want", 2, 0, "root"),
        word(text, "Tom", 3, 2, "obj", "PROPN"),
        word(text, "organize", 4, 2, "xcomp"),
        word(text, "teams", 5, 4, "obj", "NOUN"),
    ]
    e = D.observe_dag([words], text)["events"][0]
    assert (e["actor"], e["conceiver"]) == ("other", "self")
    assert e["subject_origin"] == "object_control_candidate"
    z = [[1, 1, 1, -1]]
    assert D.readout(z, [e], "desire", typed_owner=True)[0][0] == 0
    assert D.readout(z, [e], "desire_about_other", typed_owner=True)[0][0] == 1


def test_nested_modal_path_not_flattened():
    text = "I imagine wanting to organize teams."
    words = [
        word(text, "I", 1, 2, "nsubj", "PRON"),
        word(text, "imagine", 2, 0, "root"),
        word(text, "wanting", 3, 2, "xcomp", lemma="want"),
        word(text, "organize", 4, 3, "xcomp"),
        word(text, "teams", 5, 4, "obj", "NOUN"),
    ]
    e = D.observe_dag([words], text)["events"][0]
    assert e["mode_path"] == ["hypothetical", "desire"]
    assert e["status"] == "unresolved"


def test_describe_object_is_not_report_operator():
    text = "I describe teams."
    words = [
        word(text, "I", 1, 2, "nsubj", "PRON"),
        word(text, "describe", 2, 0, "root"),
        word(text, "teams", 3, 2, "obj", "NOUN"),
    ]
    graph = D.observe_dag([words], text)
    assert graph["nodes"][0]["type"] == "main_proposition"
    assert len(graph["events"]) == 1


def test_compression_cannot_recover_typed_projection():
    w = D.compression_witness()
    assert w["aggregate_difference"] == 0
    assert w["routed_difference"] == 1


def test_projection_and_dual_both_need_transport():
    result = D.transported_readout([1, 2], [[1, 0], [0, 0]], [[1, 0.1], [0, 1]])
    assert abs(result["original"] - result["transported"]) < 1e-12
    assert abs(result["original"] - result["fixed_mask"]) > 0.1
    assert result["projection_idempotence_error"] < 1e-12


def test_invalid_shapes_and_words_fail():
    with pytest.raises(ValueError):
        D.readout([[np.nan] * 4], [{}], "actual", typed_owner=True)
    with pytest.raises(ValueError):
        D.transported_readout([1], np.eye(2), np.eye(1))
    with pytest.raises(ValueError):
        D.observe_dag([[word("I", "I", 1, 9, "nsubj", "PRON")]], "I")


def test_graph_revision_replaces_context_dependency_not_just_main():
    a = {
        "author_id": "u",
        "event_id": "e",
        "record_id": "a",
        "condition": -1,
        "operation": "new",
        "graph": {"nodes": [{"node_id": "main"}, {"node_id": "context"}]},
    }
    b = {
        **a,
        "record_id": "b",
        "operation": "revise",
        "parent_id": "a",
        "graph": {"nodes": [{"node_id": "new_main"}, {"node_id": "new_context"}]},
    }
    active, nodes = D.replay_graph_records([a, b])
    assert [r["record_id"] for r in active] == ["b"]
    assert [n["node_id"] for n in nodes] == ["b/new_main", "b/new_context"]
    alias = {**a, "record_id": "alias", "operation": "alias", "parent_id": "a"}
    assert len(D.replay_graph_records([a, alias])[1]) == 2
    retract = {**a, "record_id": "r", "operation": "retract", "parent_id": "a"}
    assert D.replay_graph_records([a, retract])[1] == []


def test_state_temporal_adverb_is_not_required_to_be_sconj():
    text = "I feel uneasy when Tom must organize teams."
    words = [
        word(text, "I", 1, 2, "nsubj", "PRON"),
        word(text, "feel", 2, 0, "root"),
        word(text, "uneasy", 3, 2, "xcomp", "ADJ"),
        word(text, "when", 4, 7, "advmod", "ADV"),
        word(text, "Tom", 5, 7, "nsubj", "PROPN"),
        word(text, "must", 6, 7, "aux", "AUX"),
        word(text, "organize", 7, 2, "advcl"),
        word(text, "teams", 8, 7, "obj", "NOUN"),
    ]
    e = D.observe_dag([words], text)["events"][0]
    assert e["type"] == "state_condition"
    assert (e["conceiver"], e["actor"]) == ("self", "other")
    assert D.readout([[1, 1, -1, -1]], [e], "state", typed_owner=True)[0][0] == 1


def test_collective_subject_is_not_automatically_one_author():
    text = "We organize teams."
    words = [
        word(text, "We", 1, 2, "nsubj", "PRON", lemma="we"),
        word(text, "organize", 2, 0, "root"),
        word(text, "teams", 3, 2, "obj", "NOUN"),
    ]
    event = D.observe_dag([words], text)["events"][0]
    assert event["actor"] != "self"
    assert D.readout([[1, 1, 1, 1]], [event], "actual", typed_owner=True)[0][0] == 0


def test_natural_source_offsets_keep_raw_text_and_unknown_author(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.check_grain_math_dag_natural import sentence_bank

    text = "Tom organizes the local teams that Mary trains."
    tokens = [
        ("Tom", "tom", "PROPN", 2, "nsubj"),
        ("organizes", "organize", "VERB", 0, "root"),
        ("the", "the", "DET", 5, "det"),
        ("local", "local", "ADJ", 5, "amod"),
        ("teams", "team", "NOUN", 2, "obj"),
        ("that", "that", "PRON", 8, "obj"),
        ("Mary", "mary", "PROPN", 8, "nsubj"),
        ("trains", "train", "VERB", 5, "acl:relcl"),
        (".", ".", "PUNCT", 2, "punct"),
    ]
    lines = [
        "# newdoc id = synthetic_news",
        "# meta::genre = news",
        "# sent_id = source_1",
        "# text = " + text,
    ]
    for i, (token, lemma, upos, head, relation) in enumerate(tokens, 1):
        lines.append(f"{i}\t{token}\t{lemma}\t{upos}\t_\t_\t{head}\t{relation}\t_\t_")
    bank = sentence_bank("\n".join(lines), limit=1)
    assert bank[0]["text"] == text
    assert not bank[0]["author_id_verified"]
    graph = D.observe_dag(bank[0]["gold_words"], text)
    assert len(graph["events"]) == 1
    assert "Mary trains" in graph["events"][0]["object"]
