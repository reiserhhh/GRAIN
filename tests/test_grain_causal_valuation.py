#!/usr/bin/env python3
"""Synthetic causal-valuation checks without a model or research cache."""

import numpy as np
import pytest

from grain_research.construction import math_causal_valuation as M


def test_generic_level_is_not_diagnosticity_interaction():
    """Adding a shared cause level need not modify the action slope."""
    x = np.array([[1.0, 4.0], [3.0, 6.0]])
    slopes, level, interaction = M.behavior_effects(x)
    np.testing.assert_equal(slopes, [2, 2])
    assert level == 3
    assert interaction == 0


def test_constraint_can_change_slopes_without_halo():
    """A zero-average cause level can coexist with a changed action contrast."""
    slopes, level, interaction = M.behavior_effects(np.array([[-1, -3], [1, 3]]))
    np.testing.assert_equal(slopes, [2, 6])
    assert level == 0
    assert interaction == 4


def test_parts_preserve_opposing_causal_interactions():
    """Strong opposing components must not be dropped or called information%."""
    fields = np.array([0.3, 0.7, -0.5, 0.35])
    parts = M.cause_parts(fields)
    np.testing.assert_allclose(parts, [0.4, -0.8, 0.45])
    assert parts.sum() == pytest.approx(fields[3] - fields[0])


@pytest.mark.parametrize("mode", ["k", "v", "kv", "noop"])
def test_interchange_preserves_unaddressed_rows_and_does_not_mutate_inputs(mode):
    """Only the registered cause rows may receive the paired endpoint donor."""
    k, v = np.zeros((2, 3, 7, 4)), np.ones((2, 3, 7, 4))
    dk, dv = k + 2, v + 3
    new_k, new_v = M.install_cause(k, v, dk, dv, [4, 5], mode)
    np.testing.assert_equal(new_k[:, :, [0, 1, 2, 3, 6]], k[:, :, [0, 1, 2, 3, 6]])
    np.testing.assert_equal(new_v[:, :, [0, 1, 2, 3, 6]], v[:, :, [0, 1, 2, 3, 6]])
    np.testing.assert_equal(
        new_k[:, :, [4, 5]], (dk if mode != "v" else k)[:, :, [4, 5]]
    )
    np.testing.assert_equal(
        new_v[:, :, [4, 5]], (dv if mode != "k" else v)[:, :, [4, 5]]
    )
    assert k.max() == 0 and v.min() == 1


def test_imposed_action_is_not_evidence_but_chosen_context_can_be():
    """Zero action information does not entail zero personal context information."""
    prior = 0.4
    base_odds = prior / (1 - prior)
    assert M.posterior_odds(prior, (0.5, 0.5)) == pytest.approx(base_odds)
    assert M.posterior_odds(prior, (0.5, 0.5), (0.2, 0.8)) == pytest.approx(
        4 * base_odds
    )
    assert M.posterior_odds(prior, (0.2, 0.8)) == pytest.approx(4 * base_odds)


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [7]])
def test_invalid_cause_support_is_rejected(indices):
    """Do not silently pad, duplicate, wrap or invent causal donor addresses."""
    x = np.zeros((2, 3, 7, 4))
    with pytest.raises(ValueError):
        M.install_cause(x, x, x, x, indices, "k")


def test_incomplete_or_nonfinite_fields_are_not_filled():
    """Missing support is not an observed zero effect."""
    with pytest.raises(ValueError):
        M.behavior_effects(np.zeros((2, 1)))
    with pytest.raises(ValueError):
        M.cause_parts(np.full(4, np.nan))


def test_runner_layout_and_full_causal_transfer_account():
    """Every trait and unacting-person field uses the intended b/c/mode axes."""
    from scripts import run_grain_causal_valuation_hm134 as R

    rng = np.random.default_rng(134)
    cube = rng.normal(size=(2, 6, 2, 2, 4, 2, 7, 4))
    cube[:, :, :, :, 3] = cube[:, :, :, ::-1, 0]
    effects, parts, summaries = R.derive(cube)
    assert effects.shape == (2, 6, 4, 2, 7, 4)
    assert parts.shape == (2, 6, 2, 2, 3, 2, 7, 4)
    assert len(summaries) == 16
    for bank, family in ((0, 0), (1, 5)):
        direct = M.behavior_effects(cube[bank, family, :, :, 0])
        np.testing.assert_allclose(effects[bank, family, :2], direct[0])
        np.testing.assert_allclose(effects[bank, family, 2], direct[1])
        np.testing.assert_allclose(effects[bank, family, 3], direct[2])
    assert max(s["part_reconstruction_error"] for s in summaries) < 1e-14
    assert {name: len(rows) for name, rows in R.tables(effects, parts).items()} == {
        "natural_effects": 672,
        "selective_parts": 8064,
    }
