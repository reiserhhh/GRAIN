#!/usr/bin/env python3
"""Exact information bounds and actor-invariant source observation controls."""

import numpy as np

from grain_research.construction import math_grain_native_remainder as M
from grain_research.construction.math_observation_collision import collision_account


def test_equal_coordinates_cannot_recover_opposite_role_readings():
    account = collision_account(
        np.array([[0, 1], [0, 1], [2, 3], [2, 3]]),
        np.array([[1.0], [-1.0], [4.0], [2.0]]),
    )
    np.testing.assert_allclose(account["irreducible_mse"], 1)
    assert account["collision_rows"] == 4
    assert account["distinct_observations"] == 2


def test_injective_coordinates_have_no_finite_collision_bound():
    account = collision_account(
        np.arange(4)[:, None], np.array([[1.0], [-1.0], [4.0], [2.0]])
    )
    np.testing.assert_allclose(account["irreducible_mse"], 0)
    assert account["collision_rows"] == 0


def test_source_observation_invariance_forces_odd_component_to_remainder():
    rows = [
        {
            "variant": 0,
            "wording": w,
            "family": f,
            "condition": c,
            "polarity": b,
            "owner": o,
        }
        for w in range(2)
        for f in range(6)
        for c in (-1, 1)
        for b in (-1, 1)
        for o in (-1, 1)
    ]
    rng = np.random.default_rng(124)
    g = np.repeat(rng.normal(size=(48, 3, 4)), 2, axis=0)
    x = rng.normal(size=(96, 20))
    result = M.source_decomposition(g, x, rows)
    np.testing.assert_allclose(result["grain_part"], 0, atol=1e-12)
    np.testing.assert_allclose(result["whole"], result["remainder"], atol=1e-12)
