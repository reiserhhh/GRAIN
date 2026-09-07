#!/usr/bin/env python3
"""Numerical and finite-group tests of the query-conditioned KV identities."""

from itertools import permutations

import numpy as np
import pytest

from grain_research.vector_mechanisms.query_tilt import (
    integrated_query_response,
    pairing_component,
    query_covariance_split,
    query_jacobian,
    tilted_readout,
)


def fixture():
    rng = np.random.default_rng(10)
    return rng.normal(size=(15, 3)), rng.normal(size=(15, 2)), rng.normal(size=3)


def test_query_jacobian_is_cross_covariance():
    k, v, q = fixture()
    jac = query_jacobian(k, v, q)
    for dim in range(len(q)):
        step = np.eye(len(q))[dim] * 1e-5
        numerical = (
            tilted_readout(k, v, q + step)[0] - tilted_readout(k, v, q - step)[0]
        ) / 2e-5
        assert np.allclose(jac[:, dim], numerical, atol=1e-9)


def test_integrated_covariance_reconstructs_query_change():
    k, v, q = fixture()
    start = -q * 0.4
    observed = tilted_readout(k, v, q)[0] - tilted_readout(k, v, start)[0]
    predicted = integrated_query_response(k, v, start, q, order=32)
    assert np.allclose(predicted, observed, atol=1e-10)


def test_same_marginal_bags_can_have_opposite_readouts():
    k, v, q = np.array([[-1.0], [1.0]]), np.array([[-2.0], [2.0]]), np.array([1.0])
    first = tilted_readout(k, v, q)[0]
    second = tilted_readout(k, v[::-1], q)[0]
    assert np.allclose(first, -second)
    assert abs(first[0]) > 1.5


def test_pairing_removal_equals_complete_permutation_average():
    v = np.array([[1.0, 4.0], [-2.0, 0.0], [3.0, 2.0], [8.0, 9.0]])
    p = np.array([0.05, 0.5, 0.15, 0.3])
    average = np.mean(
        [p @ v[list(order) + [3]] for order in permutations(range(3))], axis=0
    )
    assert np.allclose(p @ v - pairing_component(p, v, [[0, 1, 2]]), average)


def test_total_cross_covariance_split():
    k, v, q = fixture()
    within, between = query_covariance_split(k, v, q, np.arange(len(k)) % 3)
    assert np.allclose(within + between, query_jacobian(k, v, q), atol=1e-12)


def test_pairing_disappears_for_uniform_within_group_weights():
    p, v = np.array([0.1, 0.1, 0.4, 0.4]), np.arange(8).reshape(4, 2)
    assert np.allclose(pairing_component(p, v, [[0, 1], [2, 3]]), 0)


def test_overlapping_groups_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        pairing_component(np.array([0.3, 0.7]), np.ones((2, 2)), [[0, 1], [1]])


def test_group_integrals_recover_pairing_with_native_group_mass_fixed():
    k, v, q = fixture()
    _, p = tilted_readout(k, v, q)
    groups = [list(range(5)), list(range(5, 11))]
    response = sum(
        p[ix].sum()
        * integrated_query_response(k[ix], v[ix], np.zeros_like(q), q, order=32)
        for ix in groups
    )
    assert np.allclose(response, pairing_component(p, v, groups), atol=1e-10)
