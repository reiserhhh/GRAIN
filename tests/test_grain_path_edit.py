#!/usr/bin/env python3
"""Finite source/edit invariants; no empirical label or neural dependencies."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_path_edit as M


def record():
    """Supply one complete numeric twelve-event addressed observation."""
    return {
        "author_key": "synthetic",
        "events": [
            {
                "event_id": str(i),
                "timestamp": float(i),
                "position": i + 24,
                "cell": i % 8,
            }
            for i in range(12)
        ],
        "edits": {"support": ["2", "4", "6", "8"], "counter": ["1", "3", "5", "7"]},
    }


def test_complete_gap_support():
    """All gap choices come from deletion topology, not a selected grid."""
    gaps = M.gap_patterns()
    assert gaps.shape == (15, 2)
    assert np.all(gaps.sum(1) <= 6)
    assert M.gap_patterns(0).tolist() == [[1, 1]]


def test_source_law_literal_counts():
    """Full profile and gap weighting agree with independent direct counts."""
    rng = np.random.default_rng(237)
    paths = rng.integers(8, size=(12, 2, 18))
    y = rng.integers(2, size=(12, 5))
    law = M.compile_source(paths, y)
    for i, (d1, d2) in enumerate(law["gaps"]):
        direct = np.zeros((32, 8, 8, 8))
        for u, windows in enumerate(paths):
            code = sum(int(v) * 2**q for q, v in enumerate(y[u]))
            for x in windows:
                for t in range(18 - d1 - d2):
                    direct[code, x[t], x[t + d1], x[t + d1 + d2]] += 1 / (
                        24 * (18 - d1 - d2)
                    )
        np.testing.assert_allclose(law["tables"][i], direct, atol=1e-14, rtol=0)


def test_edit_calculus():
    """Every signed component reproduces literal old/new field means."""
    rng = np.random.default_rng(3)
    fields = rng.random((15, 5, 8, 8, 8))
    prior = rng.random(5)
    r = M.edit_accounts(fields, M.gap_patterns(), prior, record())
    for arm in r["arms"].values():
        parts = np.array(arm["parts"])
        np.testing.assert_allclose(
            parts[:3].sum(0), np.array(arm["blind"]) - r["full"], atol=1e-14
        )
        np.testing.assert_allclose(
            parts.sum(0), np.array(arm["known_gap"]) - r["full"], atol=1e-14
        )
        assert arm["retained"] + arm["destroyed"] == 10
        assert arm["retained"] + arm["created"] == 6
    np.testing.assert_allclose(
        np.sum(r["part_actions"], 0), r["known_gap_action"], atol=1e-14
    )


def test_constant_null_and_cancellation():
    """A constant conditional field cannot create a total deletion effect."""
    fields = np.full((15, 5, 8, 8, 8), 0.7)
    r = M.edit_accounts(fields, M.gap_patterns(), np.full(5, 0.3), record())
    np.testing.assert_allclose(r["known_gap_action"], 0, atol=1e-14)
    for arm in r["arms"].values():
        assert np.max(abs(np.asarray(arm["parts"][0]))) > 0
        np.testing.assert_allclose(np.sum(arm["parts"], 0), 0, atol=1e-14)


def test_unknown_cell_stays_unobserved():
    """Unsupported queries are not partially averaged or converted to zero."""
    fields = np.full((15, 5, 8, 8, 8), 0.5)
    fields[0, :, 0, 1, 2] = np.nan
    r = M.edit_accounts(fields, M.gap_patterns(), np.full(5, 0.5), record())
    assert np.isnan(r["full"]).all()
    assert not r["arms"]["support"]["full_supported"]


@pytest.mark.parametrize("defect", ["repeat", "label", "position", "time", "overlap"])
def test_binding_guards(defect):
    """No duplicate events, hidden outcomes, invented order or ambiguous edits."""
    r = copy.deepcopy(record())
    if defect == "repeat":
        r["events"][1]["event_id"] = "0"
    elif defect == "label":
        r["target"] = 1
    elif defect == "position":
        r["events"][2]["position"] = 99
    elif defect == "time":
        r["events"][2]["timestamp"] = 0
    else:
        r["edits"]["counter"][0] = "2"
    with pytest.raises(ValueError):
        M.validate_record(r)


def test_zero_joint_target_null():
    """Duplicate identical paths at all32 profiles give no target excess."""
    path = np.tile(np.arange(8), 3)[:18]
    paths = np.tile(path, (32, 2, 1))
    law = M.compile_source(paths, M.profile_bits())
    finite = np.isfinite(law["fields"])
    np.testing.assert_allclose(law["fields"][finite], 0.5, atol=1e-14)
