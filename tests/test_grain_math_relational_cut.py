#!/usr/bin/env python3
"""Synthetic tests of complete cuts, owner differences and mixed observation laws."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from grain_research.construction import math_relational_cut as M


def row(direction=0):
    """Build a character-tokenizer fixture with intentional name changes only."""
    high, low = ("A", "B") if direction == 0 else ("B", "A")
    body = f"Person {high} high.\nPerson {low} low."
    text = "PREFIX\n" + body + "\nQUERY Person A?"
    start = text.index(body)
    properties = {high: [9, 14], low: [24, 28]}
    indices = {
        actor: list(range(start + a, start + b)) for actor, (a, b) in properties.items()
    }
    r = {
        "role": "source",
        "content": "target",
        "query": "target",
        "person": "A",
        "appraisal": "none",
        "order": direction,
        "behaviour": direction,
        "body": body,
        "rendered": text,
        "input_ids": [ord(c) for c in text],
        "properties": properties,
        "property_positions": indices["A"] + indices["B"],
        "property_a_count": len(indices["A"]),
        "body_endpoint": start + len(body) - 1,
    }
    encoded = {
        "input_ids": r["input_ids"],
        "offset_mapping": [(i, i + 1) for i in range(len(text))],
    }
    return r, encoded


def proof():
    """Return an exact complete partition with two exchanged owner names."""
    a, ea = row(0)
    b, eb = row(1)
    return a, b, M.prove_pair(a, b, M.partition(a, ea), M.partition(b, eb))


def test_exact_partition_and_changed_names():
    a, b, p = proof()
    assert len(p["names"]) == 2
    assert p["prefix"] + p["body"] + p["suffix"] == list(range(len(a["input_ids"])))
    assert [a["input_ids"][i] for i in p["suffix"]] == [
        b["input_ids"][i] for i in p["suffix"]
    ]
    assert p["names"] == p["changed_input_positions"]


def test_extra_input_difference_fails():
    a, b, p = proof()
    b = copy.deepcopy(b)
    b["input_ids"][0] += 1
    with pytest.raises(ValueError, match="Extra changed"):
        M.prove_pair(
            a,
            b,
            p | {"changed_input_positions": []},
            p | {"changed_input_positions": []},
        )


def test_changed_suffix_fails():
    a, b, _ = proof()
    b["input_ids"][-1] += 1
    with pytest.raises(ValueError, match="suffix"):
        M.prove_pair(a, b, {}, {})


def test_changed_tokenizer_and_ambiguous_body_fail():
    a, enc = row()
    enc["input_ids"] = enc["input_ids"][:-1]
    with pytest.raises(ValueError, match="Tokenizer"):
        M.partition(a, enc)
    a, enc = row()
    a["body"] = "Person"
    with pytest.raises(ValueError, match="unique"):
        M.partition(a, enc)


@pytest.mark.parametrize("arm", M.ARMS)
def test_every_patch_keeps_outside_positions(arm):
    _, _, p = proof()
    current = np.arange(p["length"] * 5, dtype=np.float32).reshape(-1, 5)
    donor = current + 2
    positions, values = M.patch_values(p, current, donor, arm, 0, 5011)
    after = current.copy()
    after[positions] = values
    outside = sorted(set(range(p["length"])) - set(positions))
    np.testing.assert_array_equal(after[outside], current[outside])
    if arm == "full":
        np.testing.assert_array_equal(
            after[p["body"] + p["suffix"]], donor[p["body"] + p["suffix"]]
        )
    if arm == "noop":
        np.testing.assert_array_equal(after, current)


def test_random_norms_pairing_and_zero_delta():
    _, _, p = proof()
    current = np.zeros((p["length"], 7), dtype=np.float32)
    donor = current + 2
    positions, values = M.patch_values(p, current, donor, "random_names", 0, 5011)
    _, inverse = M.patch_values(p, donor, current, "random_names", 1, 5011)
    np.testing.assert_allclose(
        np.linalg.norm(values, axis=1),
        np.linalg.norm(donor[positions], axis=1),
        rtol=1e-6,
    )
    np.testing.assert_allclose(inverse - donor[positions], -values, atol=1e-6)
    _, zero = M.patch_values(p, current, current, "random_names", 0, 5011)
    np.testing.assert_array_equal(zero, current[positions])


def test_constant_additive_and_interaction_nulls():
    assert M.four_cells(3, 3, 3, 3)["interaction"] == 0
    assert M.four_cells(0, 2, 5, 7)["interaction"] == 0
    cell = M.four_cells(0, 0, 0, 1)
    assert cell["interaction"] == 1 and cell["reconstruction_error"] == 0
    # A shared endpoint can be identical while a different suffix slot carries a value.
    suffixes = np.array([[0, 3, 0], [0, 7, 0]])
    assert suffixes[0, -1] == suffixes[1, -1]
    assert suffixes[0].sum() != suffixes[1].sum()


def test_invalid_arm_and_shape_fail():
    _, _, p = proof()
    states = np.zeros((p["length"], 3))
    for arm, donor in (("unknown", states), ("body", states[:-1])):
        with pytest.raises(ValueError):
            M.patch_values(p, states, donor, arm, 0, 5011)


@pytest.mark.parametrize(
    "override",
    (
        {"in_flight": 0},
        {"status": "STOPPED"},
        {"status": "COLLECTED"},
        {"attempted_forward_calls": 384},
    ),
)
def test_lifecycle_caps(override):
    status = {"status": "PREPARED", "in_flight": None, "attempted_forward_calls": 0}
    spec = {"max_forwards": 384, "max_model_seconds": 600}
    assert not M.call_allowed(status | override, spec, 0)
    assert M.call_allowed(status, spec, 599)
    assert not M.call_allowed(status, spec, 600)
