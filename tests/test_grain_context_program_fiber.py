#!/usr/bin/env python3
"""Mechanism-discrimination examples, including worlds with no common action."""

import numpy as np
import pytest

from grain_research.construction import math_conditional_mean_fiber as M
from grain_research.construction import math_context_program_fiber as C


def test_context_mass_and_association_survive_even_when_global_mean_collides():
    d = M.conditional_contrasts(2)
    g = np.tile(np.array([1, -1, -1, 1])[:, None], (2, 1))
    global_fiber = C.constraint_fiber(g, d)
    contextual = C.context_account_features(g, np.repeat([0, 1], 4))
    assert global_fiber["nullity"] == 1
    assert C.constraint_fiber(contextual, d)["nullity"] == 0
    w = M.positive_witness(np.repeat([1, -1], 4) * g[:, 0], d, global_fiber["basis"])
    np.testing.assert_allclose((w["plus"] - w["minus"]) @ g, 0, atol=1e-14)
    assert np.max(abs((w["plus"] - w["minus"]) @ contextual)) > 0
    np.testing.assert_allclose(contextual[:, [0, 2]].sum(1), 1)


def test_agreement_removes_a_pure_observer_difference():
    d = M.conditional_contrasts(1)
    g = np.array([[b, c] for b in (-1, 1) for c in (-1, 1)], float)
    y = np.zeros((4, 2, 2, 7, 2))
    joint = g[:, 0] * g[:, 1]
    y[:, 0, 0, 0, 0], y[:, 1, 0, 0, 0] = joint, -joint
    extra, _, _ = C.observer_contract(y, "A_traits_score")
    assert C.constraint_fiber(g, d)["nullity"] == 1
    assert C.constraint_fiber(np.column_stack([g, extra]), d)["nullity"] == 0


def test_common_action_can_survive_with_unchanged_fact_in_both_programs():
    d = M.conditional_contrasts(1)
    g = np.array([[b, c] for b in (-1, 1) for c in (-1, 1)], float)
    y = np.zeros((4, 2, 2, 7, 2))
    joint = g[:, 0] * g[:, 1]
    y[:, 0, 0, 0, 0] = 2 * joint
    y[:, 1, 0, 0, 0] = 2 * joint + 3 * g[:, 0]
    y[:, :, 0, 6, 0] = g[:, :1]
    extra, agreed, neutral = C.observer_contract(y, "A_traits_score_fact_score")
    null = C.constraint_fiber(np.column_stack([g, extra]), d)
    w = M.positive_witness(y[:, :, 0, 0, 0].mean(1), d, null["basis"])
    change = np.einsum("i,irpau->rpau", w["plus"] - w["minus"], y)
    assert change[0, 0, 0, 0] > 0
    np.testing.assert_allclose(change[0, 0, 0, 0], change[1, 0, 0, 0])
    assert len(agreed) == 6 and len(neutral) == 2
    np.testing.assert_allclose(change[:, 0, 6, 0], 0, atol=1e-14)


def test_units_and_context_names_do_not_choose_a_different_kernel():
    rng = np.random.default_rng(1381)
    x = rng.normal(size=(32, 5))
    d = M.conditional_contrasts(8)
    left = C.constraint_fiber(x, d)
    right = C.constraint_fiber(x * np.array([0.01, 10, -5, 1, 3]), d)
    np.testing.assert_allclose(
        left["basis"] @ left["basis"].T, right["basis"] @ right["basis"].T, atol=1e-13
    )
    labels = np.repeat([2, 9], 16)
    assert C.context_account_features(x, labels).shape == (32, 12)


def test_all_contracts_and_invalid_inputs_are_explicit():
    y = np.zeros((4, 2, 2, 7, 2))
    sizes = {
        "A_traits_score": 6,
        "A_traits_score_fact_score": 8,
        "A_traits_both_units": 12,
        "A_traits_both_units_fact_both_units": 16,
        "AB_all_both_units": 28,
    }
    for name, size in sizes.items():
        assert C.observer_contract(y, name)[0].shape == (4, size)
    with pytest.raises(ValueError):
        C.observer_contract(y, "best")
    with pytest.raises(ValueError):
        C.context_account_features(np.ones((4, 1)), np.array([0, 0, np.nan, 1]))
    with pytest.raises(ValueError):
        C.constraint_fiber(np.array([[np.nan]]), np.ones((1, 1)))


def test_constant_targets_and_targets_already_observed_have_no_witness():
    rng = np.random.default_rng(1382)
    d = M.conditional_contrasts(5)
    x = rng.normal(size=(20, 2))
    null = C.constraint_fiber(x, d)
    for y in (np.ones(20), x @ np.array([2, 4])):
        assert M.positive_witness(y, d, null["basis"])["plus"] is None
