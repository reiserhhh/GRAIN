#!/usr/bin/env python3
"""Group averaging witnesses and exact permutation preservation."""

import itertools

import numpy as np
import torch

from scripts.run_grain_orbit_response import apply_permutation, orbit_split


def test_permutation_preserves_mean_covariance_and_multiset():
    h = torch.arange(36, dtype=torch.float64).reshape(9, 4)
    groups = [[0, 1, 2, 3], [5, 6, 7]]
    p = apply_permutation(h, groups, [0, 1], 731)
    assert torch.equal(p[[4, 8]], h[[4, 8]])
    for g in groups:
        assert torch.equal(p[g].sort(dim=0).values, h[g].sort(dim=0).values)
        assert torch.allclose(p[g].mean(0), h[g].mean(0))
        assert torch.allclose(p[g].T @ p[g], h[g].T @ h[g])


def test_group_mean_equals_mean_state_but_nonlinear_scores_differ():
    h = np.array([1.0, 2.0, 6.0])
    orbit = np.array(list(itertools.permutations(h)))
    collapsed = np.full(3, h.mean())
    assert np.allclose(orbit.mean(0), collapsed)
    invariant = lambda a: float(a @ a)
    values = [invariant(v) for v in orbit]
    split = orbit_split(invariant(h), invariant(collapsed), values)
    assert abs(split["order_component"]) < 1e-12
    assert split["dispersion_component"] > 1
    assert split["identity_error"] < 1e-12


def test_position_linear_score_has_order_but_no_full_group_dispersion():
    h = np.array([1.0, 2.0, 6.0])
    orbit = np.array(list(itertools.permutations(h)))
    weights = np.array([1.0, -1.0, 2.0])
    split = orbit_split(h @ weights, np.full(3, h.mean()) @ weights, orbit @ weights)
    assert abs(split["dispersion_component"]) < 1e-12
    assert abs(split["order_component"]) > 1


def test_empty_selection_is_identity():
    h = torch.randn(8, 4)
    assert torch.equal(apply_permutation(h, [[0, 1]], [], 731), h)
