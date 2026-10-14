#!/usr/bin/env python3
"""Synthetic checks of the declared observation-focus measurement object."""

import numpy as np
import pytest

from grain_research.construction import math_observer_focus as M


def test_complete_basis_reconstruction():
    """The account retains all observations, not only desired qualified effects."""
    x = np.random.default_rng(117).normal(size=(2, 2, 2, 2, 2, 6, 4))
    np.testing.assert_allclose(M.reconstruct(M.decompose(x)), x, atol=1e-14)
    np.testing.assert_array_equal(M.CHARACTERS @ M.CHARACTERS.T, 32 * np.eye(32))


def test_query_person_and_focus_are_different_objects():
    """Same-magnitude pure query and pure focus actions must not be conflated."""
    p = np.zeros((32, 1))
    p[14] = 2
    r = np.zeros((32, 1))
    r[22] = 2
    a, b = M.qualified_effect(p, 1, -1), M.qualified_effect(r, 1, -1)
    np.testing.assert_array_equal(a["actor_difference"], [2])
    np.testing.assert_array_equal(a["actor_common"], [0])
    np.testing.assert_array_equal(b["actor_difference"], [0])
    np.testing.assert_array_equal(b["actor_common"], [-2])


def test_conditional_direct_four_cell_effects():
    """Compare actual conditional b/o sums with the32-character expression."""
    x = np.random.default_rng(5).normal(size=(2, 2, 2, 2, 2, 3))
    beta = M.decompose(x)
    for ci, c in enumerate((-1, 1)):
        for ri, r in enumerate((-1, 1)):
            bo = sum(
                (2 * b - 1) * (2 * o - 1) * x[ci, b, o, :, ri] / 4
                for b in range(2)
                for o in range(2)
            )
            actual = M.qualified_effect(beta, c, r)
            np.testing.assert_allclose(
                actual["actor_difference"], (bo[1] - bo[0]) / 2, atol=1e-14
            )
            np.testing.assert_allclose(
                actual["actor_common"], (bo[1] + bo[0]) / 2, atol=1e-14
            )


def test_positive_author_law_keeps_focus():
    """An author composition cannot silently average an observation policy."""
    beta = np.zeros((32, 1))
    beta[6] = 1
    beta[22] = 3
    beta[14] = 2
    beta[30] = 4
    x = M.reconstruct(beta)
    actual = M.positive_author_integral(x)
    # Evaluate bo coefficient at each actual query-person/focus corner independently.
    expected = np.array(
        [[1 + 3 * r + 2 * p + 4 * p * r for r in (-1, 1)] for p in (-1, 1)]
    )[..., None] * (0.8 / 6)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_unknown_support_rejected():
    """One unknown corner cannot be filled with a zero mechanism effect."""
    x = np.zeros((2, 2, 2, 2, 2))
    x[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        M.decompose(x)
    with pytest.raises(ValueError):
        M.positive_author_integral(x)
