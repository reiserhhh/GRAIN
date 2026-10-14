#!/usr/bin/env python3
"""Tests of a usable measurement rather than optimized predictive accuracy."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_pair_measurement as M
from grain_research.construction import math_route_relation as R


def instrument():
    """A fixed finite source law with route AND expression dependence."""
    j = np.random.default_rng(194).uniform(0.1, 2, (32, 8, 8))
    j /= j.sum()
    pi = j.sum((1, 2))
    p = j.sum(2)[:, :, None] * j.sum(1)[:, None, :] / pi[:, None, None]
    return M.compile_instrument(
        R.make_bundle(R.three_laws(np.stack((p, j)))),
        j.sum(0),
        [0, 0],
        "frozen-test-representation",
    )


def packet(a=(0, 1, 4), b=(1, 4, 7)):
    """Numeric current evidence without an author label or native score."""
    events = []
    for w, cells in enumerate((a, b)):
        for k, cell in enumerate(cells):
            x = np.zeros(15)
            x[0], x[5] = (1 if cell & 1 else -1), (1 if cell & 2 else -1)
            events.append(
                {
                    "event_id": f"event-{w}-{k}",
                    "window": w,
                    "condition": cell // 4,
                    "timestamp": 10 * w + k,
                    "coordinates": x.tolist(),
                }
            )
    return {
        "author_key": "numeric-author",
        "regime": M.REGIME,
        "representation_id": "frozen-test-representation",
        "events": events,
    }


def test_both_distinct_decompositions_recombine_without_deleting_context():
    q = M.score(instrument(), packet())
    assert q["status"] == "SUPPORTED"
    np.testing.assert_allclose(
        q["relative"],
        q["reference_route_composition"] + q["within_route_departure"],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        q["law_change_total"],
        q["law_change_routed"] + q["law_change_within"],
        atol=1e-15,
    )
    assert np.max(np.abs(q["reference_route_composition"])) > 0
    assert np.max(np.abs(q["within_route_departure"])) > 0


def test_all_addressed_deletion_actions_equal_literal_rescoring():
    bundle, p = instrument(), packet()
    q = M.score(bundle, p)
    for i, identity in enumerate(q["event_ids"]):
        removed = p | {"events": [e for e in p["events"] if e["event_id"] != identity]}
        d = M.score(bundle, removed)["relative"] - q["relative"]
        np.testing.assert_allclose(q["event_deletion_delta"][..., i], d, atol=1e-15)


def test_paired_deletion_interaction_not_sum_of_independent_event_effects():
    bundle, p = instrument(), packet()
    q = M.score(bundle, p)
    for i in range(3):
        for j in range(3):
            ids = {q["event_ids"][i], q["event_ids"][3 + j]}
            both = p | {"events": [e for e in p["events"] if e["event_id"] not in ids]}
            literal = M.score(bundle, both)["relative"] - q["relative"]
            literal -= (
                q["event_deletion_delta"][..., i]
                + q["event_deletion_delta"][..., 3 + j]
            )
            np.testing.assert_allclose(
                literal, q["joint_deletion_interaction"][..., i, j], atol=1e-15
            )
    assert np.max(np.abs(q["joint_deletion_interaction"])) > 1e-6


def test_cobatch_and_window_order_do_not_restandardize_an_author():
    bundle, p = instrument(), packet()
    before = M.score(bundle, p)
    M.score(bundle, packet((7, 7, 7), (0, 0, 0)))
    np.testing.assert_array_equal(before["relative"], M.score(bundle, p)["relative"])
    after = M.score(bundle, p | {"events": list(reversed(p["events"]))})
    np.testing.assert_allclose(before["relative"], after["relative"], atol=1e-14, rtol=0)


def test_duplicate_evidence_and_private_outcome_fields_are_rejected():
    b, p = instrument(), packet()
    with pytest.raises(ValueError, match="Repeated"):
        M.score(b, p | {"events": p["events"] + p["events"][:1]})
    with pytest.raises(ValueError, match="Author, regime"):
        M.score(b, p | {"big5": [0, 0, 0, 0, 0]})
    bad = copy.deepcopy(p)
    bad["events"][0]["native_score"] = 0.9
    with pytest.raises(ValueError, match="labels and native"):
        M.score(b, bad)


def test_wrong_representation_regime_time_or_coordinate_shape_rejected():
    b, p = instrument(), packet()
    for key in ("regime", "representation_id"):
        with pytest.raises(ValueError, match="must match"):
            M.score(b, p | {key: "other"})
    bad = copy.deepcopy(p)
    bad["events"][0]["timestamp"] = 100
    with pytest.raises(ValueError, match="overlap"):
        M.score(b, bad)
    bad = copy.deepcopy(p)
    bad["events"][0]["coordinates"] *= 2
    with pytest.raises(ValueError):
        M.score(b, bad)


def test_missing_window_is_unavailable_not_neutral_personality():
    q = M.score(instrument(), packet((0, 1), ()))
    assert q["status"] == "MISSING_WINDOW" and q["relative"] is None


def test_reference_is_frozen_and_not_recomputed_from_current_batch():
    b = instrument()
    b["reference_reading"][0][0] += 0.1
    with pytest.raises(ValueError, match="recalibrated"):
        M.score(b, packet())


def test_constant_reading_has_no_relative_position_or_evidence_action():
    j = np.full((32, 8, 8), 1 / 2048)
    b = M.compile_instrument(
        R.make_bundle(R.three_laws(np.stack((j, j)))),
        j.sum(0),
        [0, 0],
        "frozen-test-representation",
    )
    q = M.score(b, packet())
    np.testing.assert_allclose(q["relative"], 0)
    np.testing.assert_allclose(q["event_deletion_delta"], 0)
    np.testing.assert_allclose(q["joint_deletion_interaction"], 0)
