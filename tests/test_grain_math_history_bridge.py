#!/usr/bin/env python3
"""Synthetic observable accounts; no personality labels or neural calls."""

import numpy as np
import pytest

from grain_research.construction import math_history_bridge as M
from grain_research.construction.math_cycle_capacity import cycles


def fixture():
    """Construct smooth nondegenerate primitive changes with multiple heads."""
    rng = np.random.default_rng(1000)
    low = (
        rng.normal(size=(4, 3, 7)),
        rng.normal(size=(4, 3, 7)),
        rng.uniform(size=(4, 3, 7)),
    )
    high = (
        low[0] + rng.normal(scale=0.7, size=low[0].shape),
        low[1] + rng.normal(size=low[1].shape),
        rng.uniform(size=low[2].shape),
    )
    return low, high


def test_complete_finite_accounts_and_mobius_reconstruction():
    low, high = fixture()
    y = M.cube(low, high)
    a = M.finite_accounts(y)
    np.testing.assert_allclose(
        a["orders"].sum(-1),
        np.repeat((y[..., 7] - y[..., 0])[..., None], 6, axis=-1),
        atol=1e-13,
    )
    np.testing.assert_allclose(a["shapley"].sum(-1), y[..., 7] - y[..., 0], atol=1e-13)
    for mask in range(8):
        subsets = [s for s in range(8) if s & mask == s]
        np.testing.assert_allclose(
            a["interactions"][..., subsets].sum(-1), y[..., mask], atol=1e-13
        )


def test_path_derivative_and_integral_completeness():
    low, high = fixture()
    s, eps = 0.37, 1e-5

    def at(t):
        return M.score(*(a + t * (b - a) for a, b in zip(low, high, strict=True)))

    np.testing.assert_allclose(
        M.derivative_terms(low, high, s).sum(-1),
        (at(s + eps) - at(s - eps)) / (2 * eps),
        atol=1e-9,
    )
    a16, a32 = M.integral_accounts(low, high, 16), M.integral_accounts(low, high, 32)
    np.testing.assert_allclose(a16, a32, atol=1e-12)
    np.testing.assert_allclose(a16.sum(-1), at(1) - at(0), atol=1e-12)


def test_no_change_and_pure_value_key_gate():
    low, high = fixture()
    np.testing.assert_allclose(M.integral_accounts(low, low), 0, atol=1e-13)
    for slot, factor in ((0, 1), (1, 0), (2, 2)):
        changed = list(low)
        changed[slot] = high[slot]
        a = M.integral_accounts(low, changed)
        np.testing.assert_allclose(
            a[..., [f for f in range(3) if f != factor]], 0, atol=1e-13
        )


def test_gauge_and_joint_token_permutation_invariance():
    low, high = fixture()
    shifted = tuple(
        (a[0] + np.arange(12).reshape(4, 3, 1), a[1], a[2]) for a in (low, high)
    )
    np.testing.assert_allclose(
        M.integral_accounts(*shifted), M.integral_accounts(low, high), atol=1e-12
    )
    permutation = [6, 1, 3, 0, 5, 2, 4]
    moved = tuple(tuple(a[..., permutation] for a in side) for side in (low, high))
    np.testing.assert_allclose(M.cube(*moved), M.cube(low, high), atol=1e-13)


def test_destination_reference_annihilates_closed_author_currents():
    destination = np.arange(6, dtype=float)
    reference = np.repeat(destination[None], 6, axis=0)[..., None]
    np.testing.assert_allclose(cycles(reference)[2], 0, atol=1e-13)


def test_paths_can_disagree_without_incomplete_total():
    low, high = fixture()
    a = M.finite_accounts(M.cube(low, high))
    assert np.max(np.ptp(a["orders"], axis=-2)) > 0.1
    assert np.max(np.abs(a["shapley"] - M.integral_accounts(low, high))) > 0.001


def test_invalid_support_and_probability_rejected():
    low, high = fixture()
    with pytest.raises(ValueError):
        M.cube(low, tuple(a[..., :-1] for a in high))
    with pytest.raises(ValueError):
        M.score(low[0], low[1], np.full_like(low[2], 2))
    with pytest.raises(ValueError):
        M.integral_accounts(low, high, 1)
