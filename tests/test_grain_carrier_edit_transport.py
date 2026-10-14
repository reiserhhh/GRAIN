#!/usr/bin/env python3
"""Synthetic edit-law tests; no model calls or human personality assumptions."""

import itertools

import numpy as np
import pytest

from grain_research.construction.math_carrier_edit_transport import (
    apply_increment,
    driver_coefficients,
    edit_increment,
    new_body,
)
from grain_research.construction.math_qualified_target_field import materials


def test_new_bank_complete():
    """Every source0 record has all four predeclared replacements."""
    rows = [r for r in materials() if r["bank"] == 0]
    assert len(rows) == 16
    assert all(
        "reception" in new_body(r["body"]) and "interview" in new_body(r["body"])
        for r in rows
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_conditional_and_marginal_law(axis):
    """Full edits retain mixed terms; marginal edits deliberately remove them."""
    cube = np.array(
        [[3 + 2 * e + 5 * n + 7 * e * n for n in (-1, 1)] for e in (-1, 1)], dtype=float
    )
    for e, n in itertools.product((-1, 1), repeat=2):
        actual = edit_increment(cube, e, n, axis)
        expected = -4 * e - 14 * e * n if axis == 0 else -10 * n - 14 * e * n
        assert actual == expected
        assert edit_increment(cube, e, n, axis, marginal=True) == (
            -4 * e if axis == 0 else -10 * n
        )


def test_native_flip_cocycle():
    """Sequential source edits agree with a natural double edit by telescoping."""
    rng = np.random.default_rng(7)
    cube = rng.normal(size=(2, 2, 3))
    for e, n in itertools.product((-1, 1), repeat=2):
        i, j = (e + 1) // 2, (n + 1) // 2
        x = edit_increment(cube, e, n, 0) + edit_increment(cube, -e, n, 1)
        y = edit_increment(cube, e, n, 1) + edit_increment(cube, e, -n, 0)
        np.testing.assert_allclose(x, y, atol=1e-14)
        np.testing.assert_allclose(x, cube[1 - i, 1 - j] - cube[i, j], atol=1e-14)


def test_patch_scope_and_float64_arithmetic():
    """Unselected rows remain byte-identical under source-only arithmetic."""
    base = np.arange(24, dtype=np.float32).reshape(2, 1, 4, 3)
    delta = np.full(base.shape, 0.25)
    result = apply_increment(base, delta, [1, 2])
    np.testing.assert_array_equal(result[:, :, [0, 3]], base[:, :, [0, 3]])
    np.testing.assert_array_equal(result[:, :, [1, 2]], base[:, :, [1, 2]] + 0.25)


def test_qualified_character_and_null():
    """Name-specific action is distinguished from generic level or actor term."""
    x = np.array(
        [
            [[4 + 3 * e * p + 2 * n * p + e * n for p in (-1, 1)] for n in (-1, 1)]
            for e in (-1, 1)
        ]
    )
    expected = np.array([4, 0, 0, 1, 0, 3, 2, 0])
    np.testing.assert_allclose(driver_coefficients(x), expected)


def test_invalid_support():
    """Malformed support never silently supplies a zero observation."""
    with pytest.raises(ValueError):
        edit_increment(np.zeros((3, 2)), 1, 1, 0)
    with pytest.raises(ValueError):
        apply_increment(np.zeros((1, 1, 2, 1)), np.zeros((1, 1, 2, 1)), [2])
