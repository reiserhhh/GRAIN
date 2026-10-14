#!/usr/bin/env python3
"""SOURCE design and positive-mixture estimand tests; no private data or model."""

from itertools import combinations

import numpy as np
import pytest

from grain_research.construction import math_balanced_evidence as M


def fixture():
    """Use eight actual ordered remainders of twelve synthetic events."""
    s = np.array(list(combinations(range(12), 4)), int)
    slot = M.slot_indicators(s).reshape(len(s), -1)
    rng = np.random.default_rng(250)
    a = rng.normal(size=(len(s), 3, 5)) * 0.1
    a[:, 2] = a[:, 0] - a[:, 1]
    a[:, 2, 3] += rng.normal(size=len(s))
    a[:, 0, 3] = a[:, 1, 3] + a[:, 2, 3]
    return s, slot, a, np.arange(12) % 2


def test_positive_design_balances_each_event_slot_and_other_trait_accounts():
    s, slot, a, c = fixture()
    r = M.construct(a, s, c, 2)
    d = r["weights"][0] - r["weights"][1]
    assert np.all(r["weights"] >= 0)
    assert np.allclose(r["weights"].sum(1), 1)
    assert np.max(abs(d @ slot)) < 1e-8
    assert np.max(abs(r["source_actions"][:, [0, 1, 2, 4]])) < 1e-8
    assert r["source_actions"][2, 3] > 0
    assert np.max(abs(r["duality_gap"])) < 1e-8
    assert np.all(c[s[r["support"]]].sum(1) == 2)


def test_every_event_slot_separable_reader_is_annihilated():
    s, slot, a, c = fixture()
    r = M.construct(a, s, c, 2)
    nuisance = np.random.default_rng(20).normal(size=(96, 11))
    y = 10 + slot @ nuisance
    assert np.max(abs((r["weights"][0] - r["weights"][1]) @ y)) < 1e-8


def test_affine_source_target_has_zero_identifiable_gap_not_rescued():
    s, slot, a, c = fixture()
    a[:, 2, 3] = 1 + slot @ np.arange(96) / 100
    a[:, 0, 3] = a[:, 1, 3] + a[:, 2, 3]
    r = M.construct(a, s, c, 2)
    assert abs(r["source_actions"][2, 3]) < 1e-8


def test_pair_interaction_survives_same_event_slot_marginals():
    s, slot, a, c = fixture()
    inclusion = slot.reshape(len(s), 12, 8).sum(2)
    a[:, 2, 3] = inclusion[:, 3] * inclusion[:, 8]
    a[:, 0, 3] = a[:, 1, 3] + a[:, 2, 3]
    r = M.construct(a, s, c, 2)
    assert r["source_actions"][2, 3] > 0.01


def test_no_missing_constraint_or_condition_fallback():
    s, _, a, c = fixture()
    with pytest.raises(ValueError):
        M.construct(a[..., :4], s, c, 2)
    with pytest.raises(ValueError):
        M.construct(a, s, c, 5)
    with pytest.raises(ValueError):
        M.slot_indicators(s[:, ::-1])


def test_mixture_is_mean_of_native_log_odds_not_pooled_probability():
    p = np.array([[0.75, 0.25], [0.25, 0.75]])
    x = np.zeros((2, 2, 10))
    x[0, :, 5:] = np.array([1, 2])[:, None]
    x[1, :, 5:] = np.array([3, -1])[:, None]
    means, delta, _ = M.mixture_action(p, x)
    assert np.allclose(means, [[1.5, 1.25], [2.5, -0.25]])
    assert np.allclose(delta, [-1, 1.5])
    shifted = x + np.arange(4).reshape(2, 2, 1)
    assert np.allclose(delta, M.mixture_action(p, shifted)[1])


def test_identical_arms_are_exact_zero_and_negative_weights_rejected():
    p = np.array([[0.5, 0.5], [0.5, 0.5]])
    x = np.random.default_rng(2).normal(size=(2, 2, 10))
    assert np.array_equal(M.mixture_action(p, x)[1], [0, 0])
    with pytest.raises(ValueError):
        M.mixture_action(np.array([[1.1, -0.1], [0.5, 0.5]]), x)
