#!/usr/bin/env python3
"""Enumerated finite-group witnesses for the proposed curvature contraction."""

import itertools

import numpy as np
import pytest

from grain_research.vector_mechanisms.orbit_geometry import (
    mean_null_project,
    permutation_covariance_action,
)


def test_matrix_free_covariance_equals_full_group_enumeration():
    rng = np.random.default_rng(712)
    r = mean_null_project(rng.normal(size=(4, 3)))
    orbit = np.array([r[list(p)].ravel() for p in itertools.permutations(range(4))])
    covariance = orbit.T @ orbit / len(orbit)
    v = rng.normal(size=r.shape)
    result = permutation_covariance_action(r, v)
    assert np.allclose(result.ravel(), covariance @ v.ravel(), atol=1e-12)
    assert np.allclose(result.mean(0), 0, atol=1e-12)
    assert np.sum(v * result) >= -1e-12


def test_quadratic_orbit_response_is_hessian_covariance_contraction():
    rng = np.random.default_rng(71)
    h = rng.normal(size=(3, 2))
    r = mean_null_project(h)
    orbit = np.array([r[list(p)].ravel() for p in itertools.permutations(range(3))])
    covariance = orbit.T @ orbit / len(orbit)
    k = rng.normal(size=(6, 6))
    k = (k + k.T) / 2
    center = np.tile(h.mean(0), 3)
    linear = rng.normal(size=6)
    f = lambda x: float(0.5 * x @ k @ x + linear @ x)
    effect = np.mean([f(center + v) for v in orbit]) - f(center)
    assert abs(effect - 0.5 * np.trace(k @ covariance)) < 1e-12


def test_constant_comment_has_zero_operator():
    assert not permutation_covariance_action(np.zeros((1, 3)), np.ones((1, 3))).any()
    assert not permutation_covariance_action(np.zeros((4, 3)), np.ones((4, 3))).any()


def test_uncentered_states_rejected():
    with pytest.raises(ValueError, match="centered"):
        permutation_covariance_action(np.ones((3, 2)), np.zeros((3, 2)))
