#!/usr/bin/env python3
"""F01 ontology and source-ledger checks."""

from grain_research.gptpro_foundation_v3.f01 import GRAPH_NODES, OBJECT_CLASSES, mechanism_graph, object_dictionary, predecessor_ledger


def test_graph_has_required_measurement_chain():
    graph = mechanism_graph()
    ids = {node["id"] for node in graph["nodes"]}
    assert {"O", "C", "E", "T", "R", "S", "Q", "L"} <= ids
    assert [(edge["from"], edge["to"]) for edge in graph["edges"]][:5] == [("O", "C"), ("C", "E"), ("E", "T"), ("T", "R"), ("R", "S")]


def test_object_dictionary_has_seven_noninterchangeable_classes():
    rows = object_dictionary()
    assert [row["class"] for row in rows] == OBJECT_CLASSES
    assert all(row["counterexample"] and row["evidence"] for row in rows)


def test_predecessor_ledger_is_bounded():
    rows = predecessor_ledger()
    assert [row["id"] for row in rows] == [f"L0{i}" for i in range(1, 9)]
    assert all(row["url"].startswith("https://") and row["not_claimed"] for row in rows)
