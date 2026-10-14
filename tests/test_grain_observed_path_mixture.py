#!/usr/bin/env python3
"""Test mixture and signed accounts without treating recurrence as psych validity."""

import numpy as np
import pytest

from grain_research.construction import math_observed_path_memory as M
from grain_research.construction import math_observed_path_mixture as D


def product(a, c):
    """Return an independent two-state triple with an equiprobable middle."""
    return np.einsum("i,j,k->ijk", [a, 1 - a], [0.5, 0.5], [c, 1 - c])


def test_author_covariance_explains_pooled_memory():
    mass = np.stack([product(0.1, 0.1), product(0.9, 0.9)] * 2)
    y = np.array([[0] * 5, [0] * 5, [1] * 5, [1] * 5])
    value = D.mixture_covariance(mass, y)
    assert np.max(abs(value["covariance_mass"])) > 0.01
    assert value["identity_error"] < 1e-15
    _, _, f = M.aggregate(mass, y)
    account = D.kl_decomposition(f, f, M.markov_table(f))
    assert account["total_KL_nats"] > 0
    assert account["remaining_KL_nats"] == 0
    assert account["signed_cross_nats"] == 0


def test_covariance_requires_middle_state_weights_not_equal_author_means():
    x, z = product(0.2, 0.3), product(0.9, 0.7)
    x[:, 0] *= 1.8
    x[:, 1] *= 0.2
    z[:, 0] *= 0.2
    z[:, 1] *= 1.8
    value = D.mixture_covariance(np.stack([x, z]), np.zeros((2, 5), int))
    assert value["identity_error"] < 1e-15
    assert np.max(abs(value["covariance_mass"])) > 0


def test_remaining_memory_and_KL_cross_are_not_automatically_independent():
    matrices = np.array([[[0.5, 0.1], [0.1, 0.3]], [[0.1, 0.2], [0.2, 0.5]]])
    mass = np.repeat(matrices[:, :, None, :], 2, axis=2) / 2
    y = np.zeros((2, 5), int)
    _, ix, f = M.aggregate(mass, y)
    j = np.zeros_like(f)
    np.add.at(j, ix, M.markov_table(mass) / len(mass))
    account = D.kl_decomposition(f, j, M.markov_table(f))
    assert abs(account["signed_cross_nats"]) > 1e-6
    assert account["identity_error"] < 1e-14


def test_signed_field_cross_terms_can_reverse_total_recurrence():
    left = np.broadcast_to(
        np.array([0.1, 0.3, 0.2])[:, None, None, None, None], (3, 5, 2, 2, 2)
    )
    right = np.broadcast_to(
        np.array([0.1, 0.2, 0.3])[:, None, None, None, None], (3, 5, 2, 2, 2)
    )
    value = D.field_decomposition(left, right, np.full((2, 2, 2), 0.125))
    assert value["terms"][0, 0] > 0
    assert value["terms"][1, 0] < 0
    assert value["identity_error"] < 1e-16


def test_missing_support_and_illegal_KL_are_not_neutral():
    fields = np.full((3, 5, 2, 2, 2), np.nan)
    value = D.field_decomposition(fields, fields, np.full((2, 2, 2), 0.125))
    assert value["identity_error"] is None
    assert np.isnan(value["terms"]).all()
    with pytest.raises(ValueError):
        D.kl_decomposition([0.5, 0.5], [0.5, 0.5], [1.0, 0.0])
