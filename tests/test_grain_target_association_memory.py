#!/usr/bin/env python3
"""Finite source-state sufficiency, evidence identity and closure sanity checks."""

import numpy as np
import pytest

from grain_research.construction import math_target_association_memory as M


def world():
    """Give each binary profile several distinct event propensity laws."""
    p1 = np.tile(np.array([0.1, 0.3, 0.6, 0.9]), 2)
    return np.c_[p1, 1 - p1], np.repeat([[0.0], [1.0]], 4, axis=0)


def test_same_profile_and_event_law_different_update():
    p, y = world()
    b, _ = M.constraints(p, y, 0)
    q, _, _ = M.column_basis(b)
    r = M.witness(p, y, q, 0, [0])
    assert r["present"] and r["gap"] > 0
    np.testing.assert_allclose(b.T @ (r["plus"] - r["minus"]), 0, atol=1e-14)
    assert min(r["plus"].min(), r["minus"].min()) > 0


def test_one_step_memory_not_two_step_closed():
    p, y = world()
    b, _ = M.constraints(p, y, 1)
    q, _, _ = M.column_basis(b)
    r = M.witness(p, y, q, 0, [0, 0])
    assert r["present"] and r["gap"] > 0
    np.testing.assert_allclose(b.T @ (r["plus"] - r["minus"]), 0, atol=1e-14)
    for cell in range(2):
        s0, s1 = M.state(p, y, r["plus"]), M.state(p, y, r["minus"])
        np.testing.assert_allclose(
            s0["after"][:, cell], s1["after"][:, cell], atol=1e-14
        )


def test_covariance_update_is_direct_conditioning():
    p, y = world()
    w = np.arange(1, 9, dtype=float)
    w /= w.sum()
    s = M.state(p, y, w)
    for cell in range(2):
        np.testing.assert_allclose(
            s["after"][:, cell], M.update(p, w, cell) @ y, atol=1e-14
        )


def test_zero_support_is_missing_not_zero():
    p, y = world()
    p = np.c_[p, np.zeros(len(p))]
    s = M.state(p, y)
    assert not s["supported"][2] and np.isnan(s["after"][:, 2]).all()
    with pytest.raises(ValueError, match="zero source support"):
        M.update(p, np.full(len(p), 1 / len(p)), 2)


def test_alias_does_not_add_information_but_distinct_equal_cells_do():
    p, y = world()
    event = {"event_id": "e1", "cell": 0}
    once, _ = M.replay(p, y, [event])
    twice, trace = M.replay(p, y, [event, event])
    np.testing.assert_array_equal(once, twice)
    assert len(trace) == 1
    distinct, trace = M.replay(p, y, [event, {"event_id": "e2", "cell": 0}])
    assert len(trace) == 2 and not np.allclose(once, distinct)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        M.replay(p, y, [event, {"event_id": "e1", "cell": 1}])


def test_quotient_merges_only_identical_laws():
    p = np.array([[0.25, 0.75], [0.25, 0.75], [0.75, 0.25]])
    r = M.law_quotient(p, 4)
    assert r["closed_observation_algebra_dimension"] == 2
    assert sorted(r["sizes"].tolist()) == [1, 2]


def test_identical_laws_cannot_have_target_update_witness():
    p = np.tile([0.25, 0.75], (4, 1))
    y = np.array([[0.0], [0.0], [1.0], [1.0]])
    b, _ = M.constraints(p, y, 0)
    q, _, _ = M.column_basis(b)
    assert not M.witness(p, y, q, 0, [0])["present"]


def test_invalid_source_law_is_rejected():
    p, y = world()
    with pytest.raises(ValueError, match="normalized"):
        M.state(p * 2, y)
    with pytest.raises(ValueError, match="normalized nonnegative"):
        M.state(p, y, np.full(len(p), -1 / len(p)))
