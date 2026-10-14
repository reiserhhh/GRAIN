#!/usr/bin/env python3
"""Finite contract tests; synthetic controls are not human validation."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_observation_regime_target as M


def source(independent=False):
    """A fixed bag with unequal SOURCE order probabilities and explicit labels."""
    t = np.zeros((1, 32, 8, 8, 8))
    if independent:
        t[0, 0, 0, 0, 1] = t[0, 31, 0, 0, 1] = 0.45
        t[0, 0, 1, 0, 0] = t[0, 31, 1, 0, 0] = 0.05
    else:
        t[0, 31, 0, 0, 1], t[0, 0, 1, 0, 0] = 0.9, 0.1
    return M.disintegrate(t)


def record(cells=(0, 0, 1)):
    """Return a timestamp-attested record independent of display order."""
    return {
        "record_id": "unit",
        "author": "u",
        "regime": "raw_pandora_publication",
        "events": [
            {
                "event_id": str(i),
                "owner": "u",
                "position": i,
                "timestamp": float(i),
                "cell": c,
            }
            for i, c in enumerate(cells)
        ],
    }


def score(law, rec):
    return M.score_record(
        law["field"], law["composition"], law["prior"][0], [[1, 1]], rec
    )


def test_observed_order_law_not_uniform_permutations():
    law = source()
    assert len(M.bag_map()[0]) == 120
    assert np.all(law["composition"][0, :, 1] == 0.9)
    assert np.allclose(law["succession"][0, :, 1], 0.1)
    assert np.allclose(law["succession"][0, :, 64], -0.9)
    assert (
        M.conditional_residual_error(
            law["reference"], law["succession"], law["bag_mapping"]
        )
        < 1e-14
    )


def test_target_independent_order_null():
    law = source(True)
    assert np.nanmax(abs(law["succession"])) == 0


def test_order_only_source_and_author_relative_decomposition():
    law = source()
    a, b = score(law, record()), score(law, record((1, 0, 0)))
    assert np.allclose(a["full"], 0.1)
    assert np.allclose(b["full"], -0.9)
    assert np.allclose(a["composition"], 0)
    assert np.allclose(np.array(b["composition"]) + b["succession"], b["full"])


def test_display_permutation_does_not_change_attested_score():
    law, rec = source(), record()
    changed = copy.deepcopy(rec)
    changed["events"].reverse()
    assert score(law, rec) == score(law, changed)


@pytest.mark.parametrize(
    "kind", ["owner", "duplicate", "time", "position", "regime", "answer"]
)
def test_reject_false_qualification(kind):
    rec = record()
    if kind == "owner":
        rec["events"][1]["owner"] = "v"
    elif kind == "duplicate":
        rec["events"][1]["event_id"] = "0"
    elif kind == "time":
        rec["events"][1]["timestamp"] = 0.0
    elif kind == "position":
        rec["events"][1]["position"] = 0
    elif kind == "regime":
        rec["regime"] = "cue_mask"
    else:
        rec["answer"] = 1
    with pytest.raises(ValueError):
        M.canonical_events(rec)


def test_missing_order_not_zero_or_partial_renormalization():
    law = source()
    res = score(law, record((0, 1, 0)))
    assert res["complete"] is False
    assert res["full"] is None and res["succession"] is None
    assert np.allclose(res["composition"], 0)


def test_gap_is_not_new_adjacency():
    law, rec = source(), record()
    rec["events"][2]["position"] = 3
    res = score(law, rec)
    assert res["ledger"][0]["original_gap"] == [1, 2]
    assert res["full"] is None


def test_invalid_source_law_rejected():
    with pytest.raises(ValueError):
        M.disintegrate(np.zeros((1, 32, 8, 8, 8)))
