#!/usr/bin/env python3
"""Synthetic algebra tests for conditional-law construction and its limits."""

import numpy as np
import pytest

from grain_research.construction import math_conditional_bridge as B


def soft_world():
    """Supply a valid joint law with nonflat MODAL pairwise transports."""
    t = np.zeros((3, 3, 3))
    for a in range(3):
        t[a, a, a] += 1 / 9
        t[(a + 1) % 3, a, a] += 1 / 9
        t[(a + 1) % 3, (a + 1) % 3, a] += 1 / 9
    return t


def pairs(t):
    """Retain ordered AB, BC, AC marginals of a full table."""
    return {(0, 1): t.sum(axis=2), (1, 2): t.sum(axis=0), (0, 2): t.sum(axis=1)}


def test_permutation_convention():
    for r in B.permutations(3):
        assert np.array_equal(r.T @ r, np.eye(3))
        assert np.array_equal(r.argmax(axis=0), np.flatnonzero(r.T) % 3)


def test_pair_count_and_addresses():
    x = np.array([[0, 1], [0, 1], [2, 0]])
    j = B.empirical_pair(x)
    assert j[0, 1] == 2 / 3 and j[2, 0] == 1 / 3
    for invalid in (
        x.astype(float),
        np.array([[0, 3]]),
        np.zeros((2, 3)),
        np.zeros((0, 2)),
    ):
        with pytest.raises(ValueError):
            B.empirical_pair(invalid)


def test_observation_deconvolution():
    a = np.eye(3) * 0.6 + np.ones((3, 3)) * (0.4 / 3)
    b = np.eye(3) * 0.5 + np.ones((3, 3)) * (0.5 / 3)
    j = pairs(soft_world())[(0, 2)]
    result = B.deconvolve_pair(a @ j @ b.T, a, b)
    assert np.max(abs(result["coupling"] - j)) < 1e-12
    assert result["observation_reconstruction_error"] < 1e-12


def test_rank_loss_does_not_identify_coupling():
    with pytest.raises(ValueError, match="rank-deficient"):
        B.deconvolve_pair(np.ones((3, 3)) / 9, np.ones((3, 3)) / 3, np.eye(3))


def test_signed_inverse_is_retained_not_clipped():
    a = np.eye(3) * 0.1 + np.ones((3, 3)) * 0.3
    result = B.deconvolve_pair(np.diag([0.4, 0.3, 0.3]), a, a)
    assert result["minimum_entry"] < 0
    assert result["observation_reconstruction_error"] < 1e-12
    with pytest.raises(ValueError, match="Signed"):
        B.triangle_compatibility(
            {e: result["coupling"] for e in ((0, 1), (1, 2), (0, 2))}
        )


def test_stable_anonymous_frame_recovers_without_class_labels():
    gauges = [np.eye(3), B.permutations(3)[3], B.permutations(3)[1]]
    j = {
        (a, b): gauges[a] @ (np.eye(3) / 3) @ gauges[b].T
        for a, b in ((0, 1), (1, 2), (0, 2))
    }
    result = B.synchronize(j)
    assert result["optimum_count"] == 1
    assert all(
        np.array_equal(a, b) for a, b in zip(result["frames"][0], gauges, strict=True)
    )
    assert result["cycle_fixed_points"] == 3
    assert result["forced_consistency_cost"] == 0
    assert B.triangle_compatibility(j)["feasible"]


def test_independence_does_not_create_a_shared_class():
    j = {e: np.ones((3, 3)) / 9 for e in ((0, 1), (1, 2), (0, 2))}
    result = B.synchronize(j)
    assert result["optimum_count"] == 36
    assert result["cycle"] is None
    assert all(e["optimum_count"] == 6 for e in result["edges"].values())
    assert B.triangle_compatibility(j)["feasible"]


def test_valid_soft_response_is_not_a_flat_hard_frame():
    j = pairs(soft_world())
    result = B.synchronize(j)
    assert result["cycle_fixed_points"] == 0
    assert result["forced_consistency_cost"] == pytest.approx(1 / 3)
    assert all(
        e["match_mass_or_signed_objective"] == pytest.approx(2 / 3)
        for e in result["edges"].values()
    )
    assert B.triangle_compatibility(j)["feasible"]
    bound = B.cycle_mismatch_bound(j, result)
    assert bound["minimum_required_mass"] == pytest.approx(1)
    assert bound["sum_edge_mismatch_mass"] == pytest.approx(1)


def test_incompatible_pair_cohorts_not_one_population():
    j = {
        (0, 1): np.eye(3) / 3,
        (1, 2): np.eye(3) / 3,
        (0, 2): np.eye(3)[:, [1, 2, 0]].T / 3,
    }
    result = B.synchronize(j)
    assert not B.triangle_compatibility(j)["feasible"]
    assert result["cycle_fixed_points"] == 0
    assert B.cycle_mismatch_bound(j, result)["slack"] == pytest.approx(-1)


def test_pairwise_alias_does_not_exclude_higher_order_author_structure():
    independent = np.ones((3, 3, 3)) / 27
    parity = np.zeros_like(independent)
    for a in range(3):
        for b in range(3):
            parity[a, b, (-a - b) % 3] = 1 / 9
    for e in pairs(parity):
        assert np.max(abs(pairs(parity)[e] - pairs(independent)[e])) < 1e-12
    assert abs(parity - independent).sum() / 2 == pytest.approx(2 / 3)


@pytest.mark.parametrize("world", ["stable", "independent", "soft"])
def test_all_local_gauges_preserve_invariants(world):
    if world == "stable":
        j = {e: np.eye(3) / 3 for e in ((0, 1), (1, 2), (0, 2))}
    elif world == "independent":
        j = {e: np.ones((3, 3)) / 9 for e in ((0, 1), (1, 2), (0, 2))}
    else:
        j = pairs(soft_world())
    profiles = np.random.default_rng(4).dirichlet(np.ones(3), size=(5, 3))
    audit = B.gauge_audit(j, profiles)
    assert audit["relabeling_cases"] == 216
    assert max(audit[k] for k in audit if k.endswith("error")) < 1e-12


def test_stitching_never_averages_conditions():
    p = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    out = B.stitch_profiles(p, [np.eye(3)] * 3)
    assert out.shape == p.shape and np.array_equal(out, p)
    assert np.linalg.norm(out[:, 0] - out[:, 1]) > 0


def test_lp_witness_replays_all_marginals_but_is_not_unique():
    j = pairs(soft_world())
    witness = B.triangle_compatibility(j)
    assert witness["witness_error"] < 1e-10
    assert witness["joint_witness"].min() >= 0
    for e, marginal in pairs(witness["joint_witness"]).items():
        assert np.max(abs(j[e] - marginal)) < 1e-10
