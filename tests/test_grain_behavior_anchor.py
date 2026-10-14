#!/usr/bin/env python3
"""Meaning-key, actual-input law and complete signed interaction checks."""

import numpy as np
import pytest

from grain_research.construction import math_behavior_anchor as M


def test_family_keeps_eight_actual_events_and_each_choice_has_a_fixed_slot():
    bits, removed, slots = M.fixed_family()
    assert bits.shape == (16, 4)
    assert np.all(np.diff(removed, axis=1) > 0)
    assert np.all(slots.sum((1, 2)) == 8)
    assert np.all(slots[:, 8:, 4:].sum((1, 2)) == 4)


def test_independent_public_keys_have_equal_pole_weights_not_equal_item_weights():
    rows = [{"target": "C", "pole": "high"}] * 2 + [{"target": "C", "pole": "low"}]
    d = M.public_directions([[2, 0], [0, 2], [-1, -1]], rows, ["C"])
    assert np.allclose(d, [[2**-0.5, 2**-0.5]])


def test_public_key_sign_reversal_reverses_the_direction_without_fit():
    rows = [{"target": "C", "pole": "high"}, {"target": "C", "pole": "low"}]
    h = [[2, 3], [0, 1]]
    a = M.public_directions(h, rows, ["C"])
    reverse = [{"target": "C", "pole": "low"}, {"target": "C", "pole": "high"}]
    assert np.allclose(a, -M.public_directions(h, reverse, ["C"]))
    with pytest.raises(ValueError):
        M.public_directions([[1, 1], [1, 1]], rows, ["C"])


def test_positive_design_holds_each_event_slot_rival_mean_and_entire_count_law():
    bits, _, slots = M.fixed_family()
    signs = 1 - 2 * bits
    scores = np.zeros((16, 5))
    scores[:, 3] = signs[:, 0] * signs[:, 1]
    scores[:, 0] = signs[:, 2] * signs[:, 3]
    result = M.balanced_design(scores, np.zeros(12, int))
    nu = result["weights"][0] - result["weights"][1]
    assert np.all(result["weights"] >= 0)
    assert np.allclose(result["weights"].sum(1), 1)
    assert np.max(abs(nu @ slots.reshape(16, -1))) < 1e-8
    assert np.max(abs(result["source_action"][[0, 1, 2, 4]])) < 1e-8
    assert result["source_action"][3] == pytest.approx(2)
    assert np.allclose(*result["count_distribution"])


def test_additive_source_cannot_make_a_joint_direction_and_is_not_rescued():
    bits, _, _ = M.fixed_family()
    scores = np.zeros((16, 5))
    scores[:, 3] = 1 + bits @ np.array([1, 2, 3, 4])
    result = M.balanced_design(scores, np.arange(12) % 2)
    assert abs(result["source_action"][3]) < 1e-8


def test_rival_copy_can_annihilate_the_target_even_if_it_is_nonlinear():
    bits, _, _ = M.fixed_family()
    scores = np.zeros((16, 5))
    scores[:, 0] = scores[:, 3] = (1 - 2 * bits[:, 0]) * (1 - 2 * bits[:, 1])
    result = M.balanced_design(scores, np.zeros(12, int))
    assert abs(result["source_action"][3]) < 1e-8


def test_walsh_signed_accounts_reconstruct_every_response_without_order_deletion():
    rng = np.random.default_rng(251)
    y = rng.normal(size=(16, 5))
    p = rng.dirichlet(np.ones(16), size=2)
    a = M.walsh_accounts(y, p)
    assert np.allclose(a["characters"].T @ a["characters"], 16 * np.eye(16))
    assert np.allclose(a["reconstruction"], y)
    assert np.allclose(a["action"], (p[0] - p[1]) @ y)


def test_separable_readers_have_zero_action_but_interactions_remain():
    bits, _, slots = M.fixed_family()
    scores = np.zeros((16, 5))
    scores[:, 3] = (1 - 2 * bits[:, 0]) * (1 - 2 * bits[:, 1])
    r = M.balanced_design(scores, np.zeros(12, int))
    y = 1 + slots.reshape(16, -1) @ np.arange(96)
    assert abs(M.walsh_accounts(y, r["weights"])["action"][0]) < 1e-8
    assert M.walsh_accounts(scores[:, 3], r["weights"])["by_order"][
        2, 0
    ] == pytest.approx(2)


def test_missing_family_or_invalid_condition_is_an_error_not_zero_evidence():
    with pytest.raises(ValueError):
        M.balanced_design(np.zeros((15, 5)), np.zeros(12, int))
    with pytest.raises(ValueError):
        M.balanced_design(np.zeros((16, 5)), np.zeros(11, int))
    with pytest.raises(ValueError):
        M.walsh_accounts(np.zeros(15), np.ones((2, 16)) / 16)
