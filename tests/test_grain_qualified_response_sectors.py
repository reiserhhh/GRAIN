#!/usr/bin/env python3
"""Four-sector completeness controls, not target-interpretation tests."""

import itertools

import numpy as np
import pytest

from scripts.analyze_grain_qualified_response_sectors_hm105 import sectors


def test_four_sector_reconstruction_and_orthogonality():
    x = np.random.default_rng(105).normal(size=(6, 6))
    parts = sectors(x)
    np.testing.assert_allclose(sum(parts.values()), x, atol=1e-14)
    for a, b in itertools.combinations(parts.values(), 2):
        assert abs(np.vdot(a, b)) < 1e-13
    np.testing.assert_allclose(sum(np.sum(a**2) for a in parts.values()), np.sum(x**2))


def test_coupled_sector_is_not_just_generic_polarity():
    x = np.ones((6, 6)) * 3 + np.arange(6)[:, None] + np.arange(6)[None, :]
    np.testing.assert_allclose(sectors(x)["coupled"], 0, atol=1e-14)
    x += (np.arange(6) - 2.5)[:, None] * (np.arange(6) - 2.5)[None, :]
    assert np.linalg.norm(sectors(x)["coupled"]) > 0


def test_missing_cells_are_not_zero():
    with pytest.raises(ValueError):
        sectors(np.array([[1, np.nan], [2, 3]]))
