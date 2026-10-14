#!/usr/bin/env python3
"""Synthetic controls for author joins, not human personality validation."""

import numpy as np
import pytest

from grain_research.construction import math_author_join as J
from grain_research.construction import math_behavior_predictive as B
from scripts.run_grain_observed_family_hm170 import ROOT, read


def bank():
    """Use the unchanged complete public behavior key."""
    return B.item_bank(read(ROOT / "configs/grain_behavior_anchor_hm251.json"))


def test_additive_zero_and_bilinear_recovery():
    rng = np.random.default_rng(263)
    x, z, a, b = rng.normal(size=(4, 50))
    values = np.stack([x + a, x + b, z + a, z + b])
    np.testing.assert_allclose(J.mixed_difference(values), 0, atol=1e-15)
    values = np.stack([x * a, x * b, z * a, z * b])
    np.testing.assert_allclose(J.mixed_difference(values), (x - z) * (a - b))


def test_common_support_is_not_balanced_direction():
    chosen = np.zeros((3, 2, 4, 50))
    chosen[:, :, 0] = 2
    result = J.interaction_accounts(
        chosen, np.zeros_like(chosen), np.full(50, 5), bank()
    )
    np.testing.assert_allclose(result["sum"]["balanced_key"], 0)
    np.testing.assert_allclose(result["sum"]["common_support"], 2)
    r = result["sum"]
    np.testing.assert_allclose(
        r["equal_item_key"], r["imbalance"] * r["common_support"]
    )


def test_order_and_signed_normalizer_preserved():
    chosen = np.zeros((2, 2, 4, 50))
    part = np.zeros_like(chosen)
    keys = np.array([r["pole"] for r in bank()])
    chosen[:, 0, 0], chosen[:, 1, 0] = keys, -keys
    part[:, :, 0] = 3
    r = J.interaction_accounts(chosen, part, np.full(50, 5), bank())["sum"]
    np.testing.assert_allclose(r["order_even"], 0)
    np.testing.assert_allclose(r["order_odd"], 1)
    np.testing.assert_allclose(
        r["item_interaction"], r["chosen_interaction"] - r["normalizer_interaction"]
    )
    np.testing.assert_allclose(r["common_support"], -3)


def test_matching_is_deranged_and_label_free():
    vectors = np.arange(1, 1 + 4 * 2 * 2 * 3, dtype=float).reshape(4, 2, 2, 3)
    lengths = np.full((4, 2, 2), 20)
    conditions = np.zeros_like(lengths)
    donors, events, costs = J.joint_matching(
        vectors, lengths, conditions, np.zeros((4, 2), int)
    )
    assert sorted(donors.tolist()) == list(range(4))
    assert np.all(donors != np.arange(4))
    assert events.shape == (4, 2) and np.isfinite(costs).all()


def test_no_scope_reduction_on_missing_matching():
    vectors = np.ones((2, 2, 1, 3))
    lengths = np.full((2, 2, 1), 20)
    conditions = np.array([[[0], [0]], [[1], [1]]])
    with pytest.raises(ValueError):
        J.joint_matching(vectors, lengths, conditions, np.zeros((2, 2), int))


def test_source_contact_swap_and_component_linearity():
    rng = np.random.default_rng(263)
    x, y = rng.normal(size=(7, 2, 5)), rng.normal(size=(7, 5))
    c = J.source_contacts(x, y)
    np.testing.assert_allclose(J.source_contacts(x[:, ::-1], y), c.transpose(1, 0, 2))
    np.testing.assert_allclose(J.source_contacts(-x, y), c)
    for p in range(5):
        for r in range(5):
            for q in range(5):
                assert c[p, r, q] == pytest.approx(
                    np.mean(x[:, 0, p] * x[:, 1, r] * y[:, q])
                )


def test_nonfinite_and_incomplete_rejected():
    with pytest.raises(ValueError):
        J.mixed_difference(np.full((4, 3), np.nan))
    with pytest.raises(ValueError):
        J.interaction_accounts(
            np.zeros((2, 2, 3, 50)), np.zeros((2, 2, 3, 50)), np.full(50, 5), bank()
        )
