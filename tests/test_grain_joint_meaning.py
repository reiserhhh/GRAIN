#!/usr/bin/env python3
"""Test relation assignment separately from singleton and pooled pair evidence."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_joint_meaning as M


def source():
    """Two target profiles with three heterogeneous normalized authors each."""
    y = np.repeat([[0] * 5, [1] * 5], 3, axis=0)
    p = np.array(
        [[10 + i, 18 - i, 7 + i % 3, 9 - i % 3, 4, 6, 8, 10] for i in range(6)], float
    )
    p /= p.sum(1, keepdims=True)
    return np.stack([p, p], axis=1), y


def test_ipf_preserves_all_two_way_margins():
    t = np.random.default_rng(177).uniform(0.2, 2, (3, 4, 4))
    t /= t.sum()
    h, receipt = M.homogeneous(t)
    for a, b in zip(M.margins(t), M.margins(h), strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
    assert receipt["sweeps"] < 10000


def test_homogeneous_relation_has_no_profile_residual():
    rng = np.random.default_rng(17)
    t = (
        rng.uniform(0.2, 2, (3, 4, 1))
        * rng.uniform(0.2, 2, (3, 1, 4))
        * rng.uniform(0.2, 2, (1, 4, 4))
    )
    t /= t.sum()
    h, _ = M.homogeneous(t)
    np.testing.assert_allclose(h, t, rtol=0, atol=1e-9)


def test_profile_xor_carries_target_meaning_despite_equal_all_lower_margins():
    t = np.array([[[0.20, 0.05], [0.05, 0.20]], [[0.05, 0.20], [0.20, 0.05]]])
    h, _ = M.homogeneous(t)
    np.testing.assert_allclose(h, 0.125, atol=1e-12)
    np.testing.assert_allclose(t.sum(0), h.sum(0), atol=1e-12)
    delta = t[1] / t.sum(0) - h[1] / h.sum(0)
    np.testing.assert_allclose(delta, [[-0.3, 0.3], [0.3, -0.3]], atol=1e-12)


def test_empty_and_structural_zero_support_is_not_smoothed():
    h, receipt = M.homogeneous(np.zeros((2, 4, 4)))
    assert receipt["sweeps"] == 0 and not h.any()
    t = np.zeros((2, 4, 4))
    t[:, 0, 0] = [0.2, 0.8]
    h, _ = M.homogeneous(t)
    np.testing.assert_array_equal(h, t)


def test_fixed_solver_failure_is_not_silently_rescued():
    t = np.random.default_rng(7).uniform(0.1, 2, (3, 4, 4))
    with pytest.raises(ValueError, match="budget exhausted"):
        M.homogeneous(t, tolerance=1e-15, max_sweeps=1)


def test_all_profiles_choice_blocks_and_singletons_preserved():
    p, y = source()
    profiles, tables, _ = M.construct(p, y)
    for z0 in range(2):
        for z1 in range(2):
            s0, s1 = slice(4 * z0, 4 * z0 + 4), slice(4 * z1, 4 * z1 + 4)
            for k in (0, 1):
                for axis in (1, 2):
                    np.testing.assert_allclose(
                        tables[k, :, s0, s1].sum(axis),
                        tables[2, :, s0, s1].sum(axis),
                        atol=1e-10,
                    )
            np.testing.assert_allclose(
                tables[1, :, s0, s1].sum(0), tables[2, :, s0, s1].sum(0), atol=1e-10
            )
    laws = M.target_laws(profiles, tables)
    for axis in (-1, -2):
        for k in (0, 1):
            np.testing.assert_allclose(laws[k].sum(axis), laws[2].sum(axis), atol=2e-10)


def test_exact_posterior_account_retains_finite_margin_residual():
    p, y = source()
    c, t, _ = M.construct(p, y)
    law = M.target_laws(c, t)
    f, d = M.fields(law), law[2] - law[1]
    account = (d[:, 1] - f[1] * d.sum(1)) / law[2].sum(1)
    np.testing.assert_allclose(account, f[2] - f[1], atol=3e-16)


def test_fixed_reference_reports_all_targets():
    p, y = source()
    a, null, tail, receipt = M.matched_reference(p, y, 3, 177, 1e-10, 10000)
    assert a.shape == tail.shape == (6,) and null.shape == (3, 6)
    assert receipt["null_max_sweeps"] < 10000
    assert ((tail > 0) & (tail <= 1)).all()


def test_bundle_tampering_and_outcome_injection_rejected():
    p, y = source()
    b = M.build_bundle(p, y)
    M.validate_bundle(b)
    bad = copy.deepcopy(b)
    bad["outcome"] = 1
    with pytest.raises(ValueError):
        M.validate_bundle(bad)
    with pytest.raises(ValueError, match="excludes outcomes"):
        M.cold_record(b, {"author_key": "x", "events": [], "target": 1})


@pytest.mark.parametrize("bad", ("duplicate", "wrong_window", "bad_cell"))
def test_cold_event_qualification_is_not_relaxed(bad):
    p, y = source()
    r = {
        "author_key": "x",
        "events": [
            {"event_id": "a", "window": 0, "cell": 1},
            {"event_id": "b", "window": 1, "cell": 2},
        ],
    }
    if bad == "duplicate":
        r["events"][1]["event_id"] = "a"
    elif bad == "wrong_window":
        r["events"][1]["window"] = 2
    else:
        r["events"][1]["cell"] = 8
    with pytest.raises(ValueError):
        M.cold_record(M.build_bundle(p, y), r)


def test_cold_score_is_pair_functional_not_independent_pair_product():
    p, y = source()
    b = M.build_bundle(p, y)
    r = {
        "author_key": "x",
        "events": [
            {"event_id": "a", "window": 0, "cell": 1},
            {"event_id": "b", "window": 1, "cell": 2},
        ],
    }
    one = M.cold_record(b, r)
    r["events"] += [
        {"event_id": "c", "window": 0, "cell": 1},
        {"event_id": "d", "window": 1, "cell": 2},
    ]
    two = M.cold_record(b, r)
    np.testing.assert_allclose(one["pair_target_readout"], two["pair_target_readout"])
    assert one["cross_pairs"] == 1 and two["cross_pairs"] == 4
