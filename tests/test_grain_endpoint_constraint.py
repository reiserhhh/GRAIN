#!/usr/bin/env python3
"""Finite witnesses for the action constraint and its non-uniqueness."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.centered_adjoint import center_groups
from grain_research.vector_mechanisms.endpoint_constraint import constrain_action


def test_constraint_and_minimum_norm_null_space():
    rng = np.random.default_rng(41)
    labels = np.repeat([0, 1], 4)
    d = center_groups(rng.normal(size=(8, 5)), labels)
    g = center_groups(rng.normal(size=d.shape), labels)
    result, info = constrain_action(d, g, labels, 0.71)
    assert np.sum(d * result) == pytest.approx(0.71)
    assert abs(info["gap_after"]) < 1e-12
    v = center_groups(rng.normal(size=d.shape), labels)
    v -= np.sum(v * d) * d / np.sum(d**2)
    assert np.sum(d * (result + v)) == pytest.approx(0.71)
    assert np.linalg.norm(result + v - g) >= np.linalg.norm(result - g)
    np.testing.assert_allclose(center_groups(result, labels), result, atol=1e-12)


def test_odd_action_keeps_even_field():
    d = np.array([[-1.0, 2.0], [1.0, -2.0]])
    g = np.array([[0.4, 0.2], [-0.4, -0.2]])
    a, _ = constrain_action(d, g, [0, 0], 0.2)
    b, _ = constrain_action(-d, g, [0, 0], -0.2)
    np.testing.assert_allclose(a, b)


def test_zero_path_cannot_carry_nonzero_action():
    z = np.zeros((2, 3))
    with pytest.raises(ValueError, match="zero path"):
        constrain_action(z, z, [0, 0], 1.0)
    result, _ = constrain_action(z, z, [0, 0], 0.0)
    np.testing.assert_array_equal(result, z)
