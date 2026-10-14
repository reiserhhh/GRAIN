#!/usr/bin/env python3
"""Synthetic tests for an author-current/observer dual measurement."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_author_current as M


def test_traffic_and_marginals_match_but_current_changes():
    p, m = (M.cycle_counts(i) / 45 for i in (1, -1))
    sp, ap, bp = M.flow_parts(p)
    sm, am, bm = M.flow_parts(m)
    np.testing.assert_array_equal(sp, sm)
    np.testing.assert_array_equal(ap, -am)
    np.testing.assert_allclose(bp, 0, atol=1e-15)
    np.testing.assert_allclose(bm, 0, atol=1e-15)
    np.testing.assert_array_equal(np.diag(p), np.diag(m))


def test_dual_account_and_reversal():
    f = np.random.default_rng(0).normal(size=(3, 3, 6))
    j = M.cycle_counts(1) / 45
    a = M.readout_account(j, f)
    b = M.readout_account(j.T, f)
    assert a["identity_error"] < 1e-14
    np.testing.assert_allclose(a["traffic"], b["traffic"])
    np.testing.assert_allclose(a["cycle"], -b["cycle"])
    np.testing.assert_allclose(a["endpoint"], 0, atol=1e-14)


def test_additive_and_reset_readers_cannot_see_stationary_current():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(3, 1, 6)) + rng.normal(size=(1, 3, 6))
    p, m = (M.cycle_counts(i) / 45 for i in (1, -1))
    np.testing.assert_allclose(M.contract(p - m, f), 0, atol=1e-14)
    reset = np.broadcast_to(rng.normal(size=(1, 3, 6)), (3, 3, 6))
    np.testing.assert_allclose(M.contract(p - m, reset), 0, atol=1e-14)


def test_gradient_reader_has_zero_cycle_action():
    a = M.flow_parts(M.cycle_counts(1) / 45)[1]
    v = np.array([1.0, -2.0, 4.0])
    f = v[:, None] - v[None]
    assert abs(M.contract(a, f)) < 1e-14


def test_two_state_stationary_current_is_zero():
    j = np.array([[0.4, 0.15], [0.15, 0.3]])
    assert np.count_nonzero(M.flow_parts(j)[1]) == 0
    a = np.array([[0, 0.2], [-0.2, 0]])
    _, cycle = M.hodge_current(a)
    np.testing.assert_allclose(cycle, 0, atol=1e-14)


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_euler_exact_bag_and_open_endpoint(direction):
    c = M.cycle_counts(direction)
    p = M.euler_path(c, 9)
    np.testing.assert_array_equal(M.path_joint(p) * 45, c)
    np.testing.assert_array_equal(np.bincount(p[:-1]), [15, 15, 15])
    open_j = M.path_joint(p[:-1])
    expected = np.zeros(3)
    expected[p[0]] += 1 / 44
    expected[p[-2]] -= 1 / 44
    np.testing.assert_allclose(open_j.sum(1) - open_j.sum(0), expected, atol=1e-15)


def test_all_joint_relabelings_preserve_scalar_not_its_signed_axis():
    f = np.random.default_rng(3).normal(size=(3, 3, 6))
    j = M.cycle_counts(1) / 45
    for p in itertools.permutations(range(3)):
        ix = np.ix_(p, p)
        np.testing.assert_allclose(M.contract(j[ix], f[ix]), M.contract(j, f))


def test_hodge_general_graph_and_endpoint_account():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(4, 4))
    a = (a - a.T) / 2
    grad, cycle = M.hodge_current(a)
    np.testing.assert_allclose(cycle.sum(1), 0, atol=1e-14)
    assert abs(np.sum(grad * cycle)) < 1e-14
    j = M.path_joint(np.array([0, 0, 1, 2, 1, 2]))
    account = M.readout_account(j, rng.normal(size=(3, 3, 6)))
    assert account["identity_error"] < 1e-14


@pytest.mark.parametrize("bad", [np.ones((2, 3)), np.eye(3), [[0, np.nan], [1, 0]]])
def test_invalid_observation_laws_rejected(bad):
    with pytest.raises(ValueError):
        M.flow_parts(bad)
