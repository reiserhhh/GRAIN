#!/usr/bin/env python3
"""Finite synthetic tests for whole-target actions rather than fitted readers."""

import numpy as np
import pytest

from grain_research.construction import math_event_anchor_field as A
from grain_research.construction import math_event_target_order as M
from grain_research.construction import math_ordered_valuation as O


def bank():
    """Small complete-source fixture with rank-valid coordinates and both labels."""
    rng = np.random.default_rng(269)
    codes = np.arange(32)
    anchors = ((codes[:, None] >> np.arange(5)) & 1).astype(float)
    return {
        "source_comments": rng.normal(size=(32, 4, 3)),
        "codes": codes,
        "anchors": anchors,
        "variance": 1.0,
        "directions": np.eye(3)[:, :2],
    }


def test_parent_weights_replay():
    """The new distribution interface must preserve every old scalar reading."""
    b, x = bank(), np.array([[0.2, -0.1, 0.3], [0.0, 0.4, -0.2]])
    for key, w in M.event_weights(b, x).items():
        old = A.score_events(b, x)[key]
        bits = (b["codes"][:, None] >> np.arange(5)) & 1
        np.testing.assert_allclose(w @ bits, old["probability"], atol=1e-14)
        np.testing.assert_allclose(w @ b["anchors"], old["anchor_mean"], atol=1e-14)


def test_ties_are_merged_without_losing_mass():
    """Tie grouping preserves a law and its strict-above-cut indicator."""
    a = np.tile(np.array([0.0, 0.4, 0.4, 1.0])[:, None], (1, 5))
    s, mass = M.atom_laws(np.array([[0.1, 0.2, 0.3, 0.4]]), a, 2)
    np.testing.assert_allclose(s, [0, 0.4, 1])
    np.testing.assert_allclose(mass, [[0.1, 0.5, 0.4]])
    np.testing.assert_allclose(mass[:, s > 0.4].sum(1), [0.4])


def test_positive_binary_can_have_negative_continuous_mean():
    """Opposite readers can reflect redistribution under an identical target."""
    s, m = np.array([0.0, 0.4, 1.0]), np.array([-0.1, 0.2, -0.1])
    a = M.ordered_account(s, m)
    assert m[s > 0.0].sum() > 0
    assert a["mean_change"] < 0
    assert a["certificate"]["status"] == "SCALE_DEPENDENT"
    np.testing.assert_allclose(a["integral"], a["mean_change"], atol=1e-14)
    np.testing.assert_allclose(
        a["positive_area"] - a["negative_area"], a["mean_change"]
    )


def test_order_certificate_bounds_all_increasing_readers():
    """Exact cone interval contains every synthetic normalized monotone reader."""
    rng = np.random.default_rng(270)
    s = np.array([0.0, 0.2, 0.4, 0.7, 1.0])
    m = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
    c = M.ordered_account(s, m)["certificate"]
    for increment in rng.dirichlet(np.ones(4), 50):
        reader = np.r_[0, np.cumsum(increment)]
        value = m @ reader
        assert c["lower"] - 1e-14 <= value <= c["upper"] + 1e-14
    np.testing.assert_allclose(
        np.sort(O.tail_actions(m))[[0, -1]], [c["lower"], c["upper"]]
    )


def test_monotone_reparameterization_preserves_order_not_mean():
    """Changing target spacing preserves the cone but can reverse a mean."""
    s, m = np.array([0.0, 0.4, 1.0]), np.array([-0.1, 0.2, -0.1])
    first, second = M.ordered_account(s, m), M.ordered_account(np.sqrt(s), m)
    assert first["mean_change"] < 0 < second["mean_change"]
    assert first["certificate"] == second["certificate"]


def test_behavior_orientation_is_genuine_probability_law():
    """Negative behaviors reverse addressed pairs instead of deleting events."""
    rng = np.random.default_rng(271)
    w = rng.dirichlet(np.ones(4), 8).reshape(4, 2, 4)
    b = np.array([1.0, -2.0, 0.0, 3.0])
    d = M.behavior_contact(w, b)
    for key in ("plus", "minus"):
        assert np.all(d[key] >= 0)
        np.testing.assert_allclose(d[key].sum(), 1)
    np.testing.assert_allclose(
        d["measure"], d["scale"] * (d["plus"] - d["minus"]), atol=1e-14
    )
    opposite = M.behavior_contact(w[:, ::-1], b)
    np.testing.assert_allclose(opposite["measure"], -d["measure"], atol=1e-14)


def test_behavior_zero_is_unobserved():
    """An absent comparator cannot be manufactured as personality agreement."""
    w = np.full((2, 2, 3), 1 / 3)
    d = M.behavior_contact(w, np.zeros(2))
    assert d["status"] == "BEHAVIOR_ZERO_UNOBSERVED"
    assert d["plus"] is None and d["minus"] is None


def test_cut_contract_rejects_wrong_old_reference():
    """Changing reference cutoffs cannot silently change the inherited target."""
    b = bank()
    assert (
        M.cut_contract(b["codes"], b["anchors"], np.full(5, 0.5))["mismatches"]
        == [0] * 5
    )
    with pytest.raises(ValueError):
        M.cut_contract(b["codes"], b["anchors"], np.ones(5))


@pytest.mark.parametrize("case", ["mass", "support", "nonfinite"])
def test_invalid_account_rejected(case):
    """Never recenter, smooth or reorder invalid caller measurements."""
    s, m = np.array([0.0, 0.5, 1.0]), np.array([-0.1, 0.2, -0.1])
    if case == "mass":
        m[0] += 0.1
    elif case == "support":
        s[1] = 1
    else:
        s[1] = np.nan
    with pytest.raises(ValueError):
        M.ordered_account(s, m)


@pytest.mark.parametrize("extra", ["label", "answer", "questionnaire"])
def test_cold_rejects_outcomes(extra):
    """Author outcomes cannot enter whole-target cold scoring."""
    with pytest.raises(ValueError):
        M.cold_record({}, {"frame_key": "x", "before": {}, "after": {}, extra: 1})


def test_cold_rejects_repeated_event():
    """Copied evidence does not make a new source action."""
    event = {"event_id": "same", "vector": [0.0, 0.0, 0.0]}
    with pytest.raises(ValueError):
        M.cold_record(bank(), {"frame_key": "x", "before": event, "after": event})


def test_cold_swap_reverses_all_target_actions():
    """Whole-target scoring preserves addressed direction and all four contracts."""
    b = bank()
    rec = {
        "frame_key": "toy",
        "before": {"event_id": "one", "vector": [0.1, 0.2, -0.1]},
        "after": {"event_id": "two", "vector": [-0.1, 0.3, 0.2]},
    }
    forward = M.cold_record(b, rec)
    backward = M.cold_record(b, rec | {"after": rec["before"], "before": rec["after"]})
    for key in forward["readings"]:
        for target in A.TARGETS:
            x, y = forward["readings"][key][target], backward["readings"][key][target]
            np.testing.assert_allclose(
                x["signed_mass"], -np.asarray(y["signed_mass"]), atol=1e-14
            )
