#!/usr/bin/env python3
"""Positive/null event-partition worlds and static narrative-binding checks."""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

from grain_research.construction import math_explicit_event_links as M
from scripts.run_grain_math_explicit_event_links import token_proof
from scripts.verify_grain_math_explicit_event_links import (
    graph_weights,
    parse_relations,
)

BEHAVIOURS = {
    "extraversion": [
        ["stayed apart", "joined others"],
        ["stayed home", "joined friends"],
    ]
}


def row(actor="A", scheme="focus_alias", order=0, syntax=0, family=0, polarity=1):
    """Create a fixed toy article independently of real model scores."""
    return M.document(
        BEHAVIOURS, "extraversion", family, polarity, order, syntax, actor, scheme
    )


@pytest.mark.parametrize(
    "actor,syntax,order", itertools.product(("A", "B"), (0, 1), (0, 1))
)
def test_constant_content_word_bag(actor, syntax, order):
    a, b = (row(actor, scheme, order, syntax) for scheme in M.SCHEMES)
    assert a["action_block"] == b["action_block"]
    assert M.word_bag(a["body"]) == M.word_bag(b["body"])
    assert parse_relations(a["declaration"]) == a["declared_relations"]


@pytest.mark.parametrize(
    "actor,scheme,order", itertools.product(("A", "B"), M.SCHEMES, (0, 1))
)
def test_actor_graph_incidence_and_alternate_formula(actor, scheme, order):
    r = row(actor, scheme, order)
    events, b, w, mention = M.incidence(r["history"])
    alternate, count = graph_weights(r)
    assert len(events) == count == (3 if actor == "A" else 4)
    assert np.allclose(w, alternate)
    assert np.allclose(w @ b, np.full(count, 1 / count))
    assert np.allclose(mention, np.full(4, 0.25))
    assert w.sum() == pytest.approx(1)


@pytest.mark.parametrize("scheme", M.SCHEMES)
def test_parent_can_be_later_in_narrative(scheme):
    r = row(scheme=scheme, order=1)
    alias = next(x for x in r["history"] if x["kind"] == "alias")
    positions = {x["record_id"]: i for i, x in enumerate(r["history"])}
    assert positions[alias["record_id"]] < positions[alias["parent_id"]]
    M.validate_history(r["history"])


@pytest.mark.parametrize(
    "field,value",
    [("parent_id", "absent"), ("actor", "B"), ("event_id", "A:K4"), ("polarity", 0)],
)
def test_invalid_alias_rejected(field, value):
    history = copy.deepcopy(row()["history"])
    alias = next(r for r in history if r["kind"] == "alias")
    alias[field] = value
    with pytest.raises(ValueError):
        M.validate_history(history)


def atom_bank(null=False, code_dependent=False):
    """Build a noiseless trait law or a law with no content signal."""
    result = {}
    for family, polarity, code in itertools.product((0, 1), (0, 1), M.ORDERS[0]):
        p = np.zeros(10)
        p[4 if null else 9 * polarity] = 1
        if code_dependent:
            p = 0.8 * p + 0.2 * np.eye(10)[int(code[1])]
        result[("extraversion", family, polarity, code)] = p
    return result


@pytest.mark.parametrize("family,polarity", itertools.product((0, 1), (0, 1)))
def test_positive_and_actor_null_worlds(family, polarity):
    atoms, v = atom_bank(), np.arange(10) / 9
    for actor in ("A", "B"):
        refs = [
            M.references(row(actor, scheme, family=family, polarity=polarity), atoms, v)
            for scheme in M.SCHEMES
        ]
        difference = (
            refs[1]["scores"]["event_code_neutral"]
            - refs[0]["scores"]["event_code_neutral"]
        )
        assert difference == pytest.approx(
            (2 * polarity - 1) / 3 if actor == "A" else 0
        )
        assert (
            refs[1]["scores"]["mention_actual_code"]
            == refs[0]["scores"]["mention_actual_code"]
        )


def test_no_signal_world_stays_null():
    atoms, v = atom_bank(null=True), np.arange(10) / 9
    for actor, scheme, order, syntax in itertools.product(
        ("A", "B"), M.SCHEMES, (0, 1), (0, 1)
    ):
        ref = M.references(row(actor, scheme, order, syntax), atoms, v)
        assert all(value == pytest.approx(4 / 9) for value in ref["scores"].values())


def test_code_conditioning_and_order_are_not_confused():
    atoms, v = atom_bank(code_dependent=True), np.arange(10) / 9
    a = M.references(row(order=0), atoms, v)
    b = M.references(row(order=1), atoms, v)
    for key in a["laws"]:
        assert np.allclose(a["laws"][key], b["laws"][key])
    assert not np.allclose(
        a["laws"]["event_actual_code"], a["laws"]["event_code_neutral"]
    )


def test_complete_binary_components_reconstruct():
    cells = list(itertools.product((0, 1), repeat=3))
    values = {
        cell: (1 + 2 * (-1) ** cell[0] + 3 * (-1) ** (cell[1] + cell[2]))
        for cell in cells
    }
    coefficients = M.character_components(values)
    assert coefficients["000"] == pytest.approx(1)
    assert coefficients["100"] == pytest.approx(2)
    assert coefficients["011"] == pytest.approx(3)
    assert sum(x * x for x in coefficients.values()) == pytest.approx(
        np.mean([x * x for x in values.values()])
    )
    with pytest.raises(ValueError):
        M.character_components({cells[0]: 1})


def test_malformed_scope_or_links_rejected():
    text = M.declaration("A", 0, "focus_alias")
    with pytest.raises(ValueError):
        parse_relations(text.replace("Person A", "Person C"))
    with pytest.raises(ValueError):
        parse_relations(text.replace("a different event from", "the same event as"))


def test_full_roster_no_duplicate_conditions():
    rows = M.documents(BEHAVIOURS, ["extraversion"])
    pairs = M.paired_roster(rows)
    assert len(rows) == 64 and len(pairs) == 32
    assert len({(p["lo"], p["hi"]) for p in pairs}) == 32
    with pytest.raises(ValueError):
        M.paired_roster(rows + [rows[0]])


def test_wrong_content_alias_rejected():
    r = row()
    records = [
        {k: v for k, v in x.items() if k not in ("event_id", "kind", "parent_id")}
        for x in r["history"]
    ]
    next(x for x in records if x["occasion_code"] == "K3")["polarity"] = 0
    with pytest.raises(ValueError):
        M.canonical_events(records, "A", "focus_alias")


def test_probability_changes_do_not_need_accuracy_threshold():
    p, q = np.eye(10)[2], np.eye(10)[7]
    low, high = (
        {"score": 2 / 9, "log_odds": -2, "probabilities": p},
        {"score": 7 / 9, "log_odds": 3, "probabilities": q},
    )
    result = M.contrast(low, high, -1)
    assert result["score"] == pytest.approx(5 / 9)
    assert result["oriented_score"] == pytest.approx(-5 / 9)
    assert result["probability_L1_half"] == pytest.approx(1)


def token_rows():
    """Make a minimal exact token-bag edit confined to two relation slots."""
    a, b = row(), row(scheme="companion_alias")
    for unit, r, ids in ((0, a, [1, 2, 3, 4, 5, 6]), (1, b, [1, 2, 4, 3, 5, 6])):
        r.update(
            unit=unit,
            input_ids=ids,
            action_positions=[0, 1],
            declaration_positions=[2, 3],
            partition={"suffix": [4, 5]},
        )
    return a, b


def test_actual_token_bag_proof():
    a, b = token_rows()
    proof = token_proof(a, b)
    assert proof["changed_positions"] == [2, 3]
    assert proof["full_token_bag_equal"] and proof["action_prefix_equal"]


@pytest.mark.parametrize("mutate", ["bag", "prefix", "suffix", "actions"])
def test_uncontrolled_token_or_content_change_rejected(mutate):
    a, b = token_rows()
    if mutate == "bag":
        b["input_ids"][2] = 99
    elif mutate == "prefix":
        b["input_ids"][0], b["input_ids"][2] = b["input_ids"][2], b["input_ids"][0]
    elif mutate == "suffix":
        b["input_ids"][4], b["input_ids"][2] = b["input_ids"][2], b["input_ids"][4]
    else:
        b["action_block"] += " changed"
    with pytest.raises(ValueError):
        token_proof(a, b)


def test_nonlinear_probability_interaction_is_not_difference_of_TV_norms():
    # Equal TV sizes can hide two opposite response directions.
    base, x, y = np.eye(3)
    d_a, d_b = x - base, y - base
    assert np.abs(d_a).sum() == np.abs(d_b).sum()
    assert np.abs(d_a - d_b).sum() / 2 == pytest.approx(1)
