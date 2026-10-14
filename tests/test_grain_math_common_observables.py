#!/usr/bin/env python3
"""Synthetic mechanism tests for material-common event and author observations."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_common_observables as M


def cube():
    """Return the complete coded support without psychological labels."""
    return np.asarray(list(itertools.product((-1, 1), repeat=3)), float).T


def test_joint_products_restore_profiles_lost_by_common_means():
    x, y, g = cube()
    f0, f1 = (
        np.column_stack((x, y, g, x * y, x * g)),
        np.column_stack((x, y, g, x * y, y * g)),
    )
    c = M.chart(f0, f1)
    assert int(c["contrast_dimension"]) == 4
    assert int(c["joint_rank"]) == 8
    p = np.array([np.ones(8) / 8 - 0.04 * y * g, np.ones(8) / 8 + 0.04 * y * g])
    observed = M.profile_moments(p, c)
    np.testing.assert_allclose(observed["means"][0], observed["means"][1], atol=1e-12)
    assert (
        np.linalg.norm(observed["joint_moments"][0] - observed["joint_moments"][1])
        > 0.01
    )
    np.testing.assert_allclose(
        observed["product_of_means"][0], observed["product_of_means"][1], atol=1e-12
    )
    np.testing.assert_allclose(observed["reconstructed"], p, atol=1e-12)


def test_affine_gauge_and_bank_exchange_preserve_common_projector():
    x, y, g = cube()
    a, b = np.column_stack((x, y, g, x * y)), np.column_stack((x, y, g, y * g))
    transform = np.array(
        [[2.0, 0.2, 0, 0], [0, 1, 0.3, 0], [0, 0, 3, 0.1], [0, 0, 0, 1]]
    )
    base = M.chart(a, b)
    transformed = M.chart(a @ transform + np.array([0.1, 0.4, -0.2, 0.3]), b)
    exchanged = M.chart(b, a)
    np.testing.assert_allclose(base["projector"], transformed["projector"], atol=1e-12)
    np.testing.assert_allclose(base["projector"], exchanged["projector"], atol=1e-12)


def test_full_event_functions_are_not_new_factors():
    c = M.chart(np.eye(8), np.eye(8) * 3)
    assert int(c["contrast_dimension"]) == 7
    assert int(c["joint_rank"]) == 8
    np.testing.assert_allclose(c["basis"].sum(0), 0, atol=1e-12)
    np.testing.assert_allclose(c["basis"].T @ c["basis"], np.eye(7), atol=1e-12)


def test_constant_only_common_observation_preserves_failure():
    x, y, _ = cube()
    c = M.chart(x[:, None], y[:, None])
    assert c["basis"].shape == (8, 0)
    assert int(c["joint_rank"]) == 1
    p = np.array([np.ones(8) / 8 + 0.04 * x])
    observed = M.profile_moments(p, c)
    np.testing.assert_allclose(
        observed["reconstructed"], np.ones((1, 8)) / 8, atol=1e-12
    )
    assert np.linalg.norm(observed["reconstructed"] - p) > 0.01


def test_separation_does_not_imply_quadratic_full_rank():
    f = np.arange(8, dtype=float)[:, None]
    c = M.chart(f, f * 2 + 5)
    assert float(c["minimum_event_separation"]) > 0
    assert int(c["joint_rank"]) == 3


def test_row_identity_must_be_carried_with_observations():
    x, y, g = cube()
    f = np.column_stack((x, y, g, x * y))
    c = M.chart(f, f)
    permutation = np.array([7, 2, 0, 4, 1, 3, 6, 5])
    moved = M.chart(f[permutation], f[permutation])
    np.testing.assert_allclose(
        moved["projector"], c["projector"][permutation][:, permutation], atol=1e-12
    )


@pytest.mark.parametrize(
    "bad", [np.array([]), np.zeros((0, 2)), np.array([[np.nan]]), np.array([[np.inf]])]
)
def test_nonfinite_or_empty_input_rejected(bad):
    with pytest.raises(ValueError):
        M.chart(bad, np.ones((8, 1)))


@pytest.mark.parametrize(
    "bad",
    [
        np.full((2, 8), 0.2),
        np.full((1, 7), 1 / 7),
        np.array([[-1.0, 2.0, 0, 0, 0, 0, 0, 0]]),
    ],
)
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        M.profile_moments(bad, M.chart(np.eye(8), np.eye(8)))


def test_unaligned_support_rejected():
    with pytest.raises(ValueError):
        M.chart(np.eye(8), np.eye(7))
