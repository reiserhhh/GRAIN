#!/usr/bin/env python3
"""Test observation-law distinctions in worlds whose mechanisms are known."""

import numpy as np
import pytest

from grain_research.construction import math_writer_choice_observation as M


def test_output_link_can_create_probability_interaction_without_logit_interaction():
    p = np.array([-1.0, 0.0, 1.0])
    ell = 0.7 + p[:, None] + 0.4 * p[None, :]
    accounts = M.link_accounts(ell)
    assert np.max(abs(accounts["q_interaction"])) > 0.01
    np.testing.assert_allclose(accounts["log_odds_interaction"], 0, atol=1e-15)
    np.testing.assert_allclose(accounts["response_remainder"], 0, atol=1e-15)
    assert accounts["q_reconstruction_error"] < 1e-15


def test_true_mixed_response_is_not_removed_as_link_curvature():
    p = np.array([-1.0, 0.0, 1.0])
    ell = 0.3 + 0.4 * p[:, None] * p[None, :]
    accounts = M.link_accounts(ell)
    assert np.max(abs(accounts["response_remainder"])) > 0.05
    assert accounts["q_reconstruction_error"] < 1e-15


def test_all_three_state_contrasts_reconstruct_and_preserve_energy():
    values = np.arange(9, dtype=float).reshape(3, 3) ** 2
    coefficients, error = M.orthogonal_accounts(values)
    assert error < 1e-12
    assert np.isclose(np.sum(coefficients**2), np.sum(values**2))


def test_polarity_and_presence_interactions_are_different_coordinates():
    p = np.array([-1.0, 0.0, 1.0])
    polarity, _ = M.orthogonal_accounts(p[:, None] * p[None, :])
    presence, _ = M.orthogonal_accounts(p[:, None] * p[None, :] ** 2)
    assert polarity[1, 1] > 1
    assert abs(presence[1, 1]) < 1e-15
    assert presence[1, 2] > 1


def test_opposite_order_effects_are_preserved_not_called_absent_signal():
    values = np.tile([-0.2, 0, 0.2], (3, 1))
    accounts = M.aligned_order_accounts(values, -values)
    np.testing.assert_array_equal(accounts["even"], np.zeros((3, 3)))
    np.testing.assert_array_equal(accounts["odd"], values)
    assert accounts["reconstruction_error"] == 0


@pytest.mark.parametrize("values", [np.zeros((2, 3)), np.full((3, 3), np.nan)])
def test_missing_or_nonfinite_tables_cannot_become_zero_observations(values):
    with pytest.raises(ValueError):
        M.nine_table(values)
