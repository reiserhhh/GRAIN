#!/usr/bin/env python3
"""Synthetic mechanism controls; no neural or private input dependencies."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_relational_operation_specificity as M
from grain_research.construction.math_relational_cut import call_allowed
from grain_research.construction.math_source_relational_transport import source_sectors

ROOT = Path(__file__).resolve().parents[1]


def test_complete_new_documents_and_shared_controls():
    spec = json.loads(
        (
            ROOT / "configs/grain_math_relational_operation_specificity_v1.json"
        ).read_text()
    )
    rows = M.documents(spec)
    assert len(rows) == 48
    assert len({r["body"] for r in rows}) == 28
    assert {r["class"] for r in rows} == set(M.CLASSES)
    for r in rows:
        assert r["body"].count("Person A ") == 1
        assert r["body"].count("Person B ") == 1
        if r["class"] == "other_report":
            assert "reported that Person C" in r["body"]
            assert "reported that Person D" in r["body"]
        if r["class"] == "roster":
            assert "included in the roster" in r["body"]


def test_pair_mapping_allows_different_absolute_lengths_not_queries():
    row = {
        "content": "e",
        "query": "e",
        "class": "own",
        "person": "A",
        "direction": 0,
        "input_ids": [8, 1, 2],
        "partition": {"suffix": [1, 2]},
    }
    donor = row | {
        "direction": 1,
        "input_ids": [8, 9, 1, 2],
        "partition": {"suffix": [2, 3]},
    }
    assert M.prove_pair(row, donor)["donor_positions"] == [2, 3]
    for change in ({"person": "B"}, {"direction": 0}, {"input_ids": [8, 9, 1, 3]}):
        with pytest.raises(ValueError):
            M.prove_pair(row, donor | change)


def test_partition_uses_actual_body_offsets():
    row = {"rendered": "p:xy?q!", "body": "xy", "input_ids": list(range(7))}
    encoded = {
        "input_ids": list(range(7)),
        "offset_mapping": [(i, i + 1) for i in range(7)],
    }
    p = M.partition(row, encoded)
    assert p["prefix"] == [0, 1] and p["body"] == [2, 3] and p["suffix"] == [4, 5, 6]
    with pytest.raises(ValueError):
        M.partition(row, encoded | {"input_ids": [8] * 7})
    with pytest.raises(ValueError):
        M.partition(row | {"body": "missing"}, encoded)


def test_invalid_orientation_stays_invalid():
    with pytest.raises(ValueError):
        M.orientation("C", 0)
    with pytest.raises(ValueError):
        M.orientation("A", 2)


@pytest.mark.parametrize("arm", M.ARMS)
def test_fixed_arms_and_unpatched_positions(arm):
    current = np.arange(24, dtype=np.float32).reshape(6, 4)
    donor = np.ones((8, 4), dtype=np.float32)
    sectors = source_sectors(np.ones((2, 4)), -2 * np.ones((2, 4)))
    entry = {
        "partition": {"suffix": [4, 5]},
        "mapping": {"donor_positions": [6, 7]},
        "person": "A",
        "direction": 0,
        "pair": 0,
        "arm": arm,
    }
    p, v = M.patch_values(entry, current, donor, sectors, 5201)
    after = current.copy()
    after[p] = v
    np.testing.assert_array_equal(after[:4], current[:4])
    if arm == "noop":
        np.testing.assert_array_equal(after, current)
    if arm == "exact_suffix":
        np.testing.assert_array_equal(v, donor[6:])
    if arm == "source_full":
        np.testing.assert_array_equal(v, current[4:] + 1)
    if arm == "wrong_person":
        np.testing.assert_array_equal(v, current[4:] - 2)


def test_invalid_arm_and_source_shape_stop():
    current = np.zeros((6, 4), np.float32)
    e = {
        "partition": {"suffix": [4, 5]},
        "mapping": {"donor_positions": [4, 5]},
        "person": "A",
        "direction": 0,
        "pair": 0,
        "arm": "unknown",
    }
    s = source_sectors(np.ones((3, 4)), np.ones((3, 4)))
    with pytest.raises(ValueError):
        M.patch_values(e, current, current, s, 5201)
    with pytest.raises(ValueError, match="shape"):
        M.patch_values(e | {"arm": "source_full"}, current, current, s, 5201)


def test_log_odds_gauge_null_and_saturation():
    z = np.array([-20.0] * 5 + [20.0] * 5)
    assert M.high_low_log_odds(z) == pytest.approx(40.0)
    assert M.high_low_log_odds(z + 1000) == pytest.approx(40.0)
    assert M.high_low_log_odds(np.zeros(10)) == 0
    # Probability is saturated, but the fixed log-odds still changes.
    zz = z + np.array([0.0] * 5 + [1.0] * 5)
    assert M.high_low_log_odds(zz) - M.high_low_log_odds(z) == pytest.approx(1.0)
    for bad in (np.zeros(9), np.full(10, np.nan)):
        with pytest.raises(ValueError):
            M.high_low_log_odds(bad)


@pytest.mark.parametrize(
    "person,direction,value", [("A", 0, -1), ("A", 1, 1), ("B", 0, 1), ("B", 1, -1)]
)
def test_orientation_not_selected_from_output(person, direction, value):
    assert M.orientation(person, direction) == value


def test_zero_natural_contrast_does_not_hide_operation():
    rows = [
        {
            "person": "A",
            "direction": 0,
            "full_change": 0.0,
            "patch_change": -0.2,
            "patch_log_odds": -1.0,
            "native_log_odds": 0.0,
        }
    ]
    s = M.summary(rows)
    assert s["signed_projection"] is None and s["native_contrast_tiny"]
    assert s["oriented_score_mean"] == pytest.approx(0.2)
    assert s["oriented_log_odds_mean"] == 1.0


def test_constant_action_world_and_empty_summary():
    r = {
        "person": "B",
        "direction": 0,
        "full_change": 0.5,
        "patch_change": 0.5,
        "patch_log_odds": 2.0,
        "native_log_odds": 0.0,
    }
    assert M.summary([r])["remaining_rmse"] == 0
    assert M.summary([r])["signed_projection"] == 1
    with pytest.raises(ValueError):
        M.summary([])


def test_no_uncertain_or_terminal_restart():
    s = {"status": "PREPARED", "in_flight": None, "attempted_forward_calls": 0}
    spec = {"max_forwards": 672, "max_model_seconds": 600}
    assert call_allowed(s, spec, 599)
    for changed in (
        {"in_flight": 10},
        {"status": "STOPPED"},
        {"status": "COLLECTED"},
        {"attempted_forward_calls": 672},
    ):
        assert not call_allowed(s | changed, spec, 1)
    assert not call_allowed(s, spec, 600)
