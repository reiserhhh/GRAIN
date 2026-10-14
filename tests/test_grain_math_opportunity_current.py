#!/usr/bin/env python3
"""Synthetic checks for opportunity/response accounting, without raw text."""

import numpy as np
import pytest

from grain_research.construction import math_opportunity_current as M


def test_fixed_condition_reset_kernel_satisfies_detailed_balance():
    for condition in range(3):
        pi = np.full(3, 0.1)
        pi[condition] = 0.8
        transition = np.tile(pi, (3, 1))
        np.testing.assert_allclose(pi @ transition, pi)
        joint = pi[:, None] * transition
        np.testing.assert_allclose(joint, joint.T)


def test_coarsening_conditions_can_fabricate_response_current():
    toy = M.toy_worlds()
    coarse = toy["forward_opportunity_only"][0] - toy["unordered_opportunity_only"][0]
    assert np.linalg.norm(M.components(coarse)[2]) > 0.1
    np.testing.assert_allclose(toy["forward_opportunity_only"][2], 0)


def test_state_relabeling_transports_all_accounts():
    order = [2, 0, 1]
    for joint in M.toy_worlds()["forward_positive_response"]:
        original = M.components(joint)
        moved = M.components(joint[np.ix_(order, order)])
        np.testing.assert_allclose(moved, original[:, order][:, :, order], atol=1e-15)


def test_environment_can_generate_cycle_without_personal_response():
    world = M.toy_worlds()
    forward = world["forward_opportunity_only"]
    reverse = world["reverse_opportunity_only"]
    assert np.linalg.norm(M.components(forward[0])[2]) > 0.1
    np.testing.assert_allclose(forward[2], 0, atol=1e-15)
    np.testing.assert_allclose(reverse[0], forward[0].T, atol=1e-15)
    np.testing.assert_allclose(
        M.components(world["unordered_opportunity_only"][0])[2], 0, atol=1e-15
    )


def test_personal_response_survives_same_opportunity():
    worlds = M.toy_worlds()
    plus, minus = (
        worlds["forward_positive_response"],
        worlds["forward_negative_response"],
    )
    np.testing.assert_allclose(plus[1], minus[1])
    np.testing.assert_allclose(plus[2], -minus[2], atol=1e-15)
    assert np.linalg.norm(M.components(plus[2])[2]) > 0.01


def test_exact_decomposition_including_nonorthogonal_cross_term():
    for j in M.toy_worlds().values():
        c = np.array([M.components(x)[2] for x in j])
        np.testing.assert_allclose(c[0], c[1] + c[2], atol=1e-15)
        np.testing.assert_allclose(
            np.sum(c[0] ** 2), np.sum(c[1:] ** 2) + 2 * np.sum(c[1] * c[2]), atol=1e-15
        )


def test_no_wrap_and_positions_preserved():
    c = M.pair_counts([[0, 1, 2, 3]], [1], 4, 2)
    assert c.sum() == 3
    assert c[1, 0, 0, 1] == 1 and c[1, 2, 2, 3] == 1
    assert c[:, :, 3, 0].sum() == 0


def test_residual_is_signed_not_a_probability():
    c = M.pair_counts([[0, 1, 2, 3]], [1], 4, 2)
    ref = np.full(c.shape, 1 / 16)
    j, q = M.reference_account(c, ref)
    assert j[2].min() < 0 and abs(j[2].sum()) < 1e-15
    np.testing.assert_allclose(j[0], j[1] + j[2])
    assert q.sum() == pytest.approx(1)


def test_fallback_is_explicit_and_reference_normalized():
    c = M.pair_counts([[0, 1, 2, 3]], [1], 4, 2)
    refs, support, fallback = M.build_references(np.stack([c] * 6), 0.5, 5)
    assert (support[0] == 0).all() and fallback[0].all()
    np.testing.assert_allclose(refs["joint"][0], refs["position"][0])
    for ref in refs.values():
        np.testing.assert_allclose(ref.sum((-2, -1)), 1)


def test_matching_never_selects_the_true_author_as_stranger():
    f = np.repeat(np.arange(12)[:, None, None], 3, axis=1)
    q = np.zeros((12, 3, 2))
    scores, _ = M.matched_author_closeness(f, q, 5)
    np.testing.assert_equal(scores, 1)


@pytest.mark.parametrize(
    "states,conditions",
    [([[0, 1, 2]], [0]), ([[0, 1, 2, 4]], [0]), ([[0, 1, 2, 3]], [2])],
)
def test_support_rejects_invalid_observations(states, conditions):
    with pytest.raises(ValueError):
        M.pair_counts(states, conditions, 4, 2)
