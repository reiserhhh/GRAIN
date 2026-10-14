#!/usr/bin/env python3
"""Exact finite moment witnesses and continuous numerical checks."""

from itertools import product

import numpy as np
import pytest

from grain_research.construction import math_role_moments as M


def test_all_binary_cells_reconstruct_qualification_and_contribution():
    for cell in product((-1, 1), repeat=4):
        z = np.array([cell], dtype=float)
        m = M.joint_moments(z)
        q = int(cell[0] == cell[2] == cell[3] == 1)
        assert M.gate_mass_from_moments(m) == q
        assert M.contribution_from_moments(m) == q * cell[1]


def test_equal_marginals_can_hide_opposite_author_evidence():
    left = np.array([[1, -1, 1, 1], [1, 1, -1, 1]])
    right = np.array([[1, 1, 1, 1], [1, -1, -1, 1]])
    assert np.array_equal(np.sort(left, axis=0), np.sort(right, axis=0))
    assert M.independent_marginal_contribution(left) == 0
    assert M.independent_marginal_contribution(right) == 0
    assert M.contribution_from_moments(M.joint_moments(left)) == -0.5
    assert M.contribution_from_moments(M.joint_moments(right)) == 0.5


def test_joint_event_permutation_preserves_binding_but_column_swap_does_not():
    z = np.array([[1, -1, 1, 1], [1, 1, -1, 1]])
    assert np.array_equal(M.joint_moments(z), M.joint_moments(z[::-1]))
    broken = z.copy()
    broken[:, 2] = broken[::-1, 2]
    assert not np.array_equal(M.joint_moments(z), M.joint_moments(broken))


def test_continuous_identity_requires_joint_moments_not_probabilities():
    rng = np.random.default_rng(4108)
    z = rng.uniform(-1, 1, (200, 4))
    z[:, 1] *= 3
    m = M.joint_moments(z)
    gate = np.prod((1 + z[:, [0, 2, 3]]) / 2, axis=1)
    assert M.contribution_from_moments(m) == pytest.approx(np.mean(z[:, 1] * gate))
    assert M.gate_mass_from_moments(m) == pytest.approx(np.mean(gate))
    assert not np.isclose(
        M.independent_marginal_contribution(z), np.mean(z[:, 1] * gate)
    )


def test_invalid_or_absent_observations_are_not_zero_scores():
    for z in (np.empty((0, 4)), np.ones((3, 3)), np.full((1, 4), np.nan)):
        with pytest.raises(ValueError):
            M.joint_moments(z)
    m = M.joint_moments(np.array([[-1, 1, 1, 1]]))
    assert M.gate_mass_from_moments(m) == 0
