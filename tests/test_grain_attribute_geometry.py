"""Finite vector expansion, mixed difference and rotation-invariance tests."""

import numpy as np
import pytest

from scripts.analyze_grain_attribute_geometry import SIGNS, describe, expand


def test_exact_world_expansion_matches_direct_linear_solve():
    rng = np.random.default_rng(12)
    worlds = {k: rng.normal(size=9) for k in SIGNS}
    coeff, error = expand(worlds)
    matrix = np.array([[1, b, r, b * r] for b, r in SIGNS])
    assert np.allclose(
        coeff, np.linalg.solve(matrix, np.array([worlds[k] for k in SIGNS]))
    )
    assert error < 1e-12
    b, r = 1, -1
    mixed = worlds[-b, -r] - worlds[-b, r] - worlds[b, -r] + worlds[b, r]
    assert np.allclose(mixed, 4 * b * r * coeff[3])


def test_geometry_is_rotation_invariant_and_degenerate_angle_is_missing():
    rng = np.random.default_rng(71)
    coeff = rng.normal(size=(4, 12))
    rotation = np.linalg.qr(rng.normal(size=(12, 12)))[0]
    first, second = describe(coeff), describe(coeff @ rotation)
    assert all(np.isclose(first[k], second[k]) for k in first)
    coeff[1] = 0
    assert np.isnan(describe(coeff)["B_R_angle_degrees"])


def test_incomplete_rectangle_is_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        expand({(-1, -1): np.zeros(2)})
