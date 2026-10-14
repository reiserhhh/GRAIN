#!/usr/bin/env python3
"""Synthetic mechanism tests; no research caches or model outputs required."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_native_access as A
from grain_research.construction import math_native_key_sectors as M


def test_four_keys_reconstruct_and_select_only_addressed_sectors():
    """All sector permissions preserve even keys and reconstruct the full array."""
    rng = np.random.default_rng(17)
    keys = rng.normal(size=(2, 2, 3, 5))
    sectors = M.key_characters(keys)
    for i, j in itertools.product(range(2), repeat=2):
        b, o = 2 * i - 1, 2 * j - 1
        np.testing.assert_allclose(M.reconstruct_key(sectors, b, o, 1, 1), keys[i, j])
        blind = M.reconstruct_key(sectors, b, o, 0, 0)
        np.testing.assert_allclose(blind, sectors[0] + b * sectors[1])
        np.testing.assert_allclose(
            M.reconstruct_key(sectors, b, o, 1, 0) - blind, o * sectors[2]
        )
        np.testing.assert_allclose(
            M.reconstruct_key(sectors, b, o, 0, 1) - blind, b * o * sectors[3]
        )


def test_generic_role_can_create_qualified_output_without_joint_keys():
    """A fixed role selector times behavioral values already yields bop."""
    keys = np.zeros((2, 2, 1))
    for i, j in itertools.product(range(2), repeat=2):
        keys[i, j] = 2 * j - 1
    sectors = M.key_characters(keys)
    assert sectors[3, 0] == 0
    cube = np.zeros((2, 2, 2))
    for i, j, k in itertools.product(range(2), repeat=3):
        b, o, p = (2 * index - 1 for index in (i, j, k))
        cube[i, j, k] = b * p * M.reconstruct_key(sectors, b, o, 1, 0)[0]
    assert A.characters(cube)[7] == 1


def test_output_parts_preserve_cancellation_and_interaction():
    """Negative finite parts and nonlinear interactions must not be dropped."""
    fields = np.array([0.1, 0.5, -0.4, 0.11])
    parts = M.intervention_parts(fields)
    np.testing.assert_allclose(parts, [0.1, 0.4, -0.5, 0.11])
    assert parts.sum() == pytest.approx(fields[3])


def test_author_account_is_an_actual_integral_not_personality_label():
    """Same marginals retain joint evidence organization and both people."""
    rng = np.random.default_rng(8)
    cube = rng.normal(size=(2, 2, 2, 3))
    beta = A.characters(cube)
    eta, direct = 0.5, []
    for p in range(2):
        delta = np.zeros(3)
        for b, o in itertools.product(range(2), repeat=2):
            delta += eta * (2 * b - 1) * (2 * o - 1) * cube[b, o, p] / 2
        direct.append(delta)
    np.testing.assert_allclose(M.author_difference(beta, eta), direct)
    np.testing.assert_allclose(M.independent_characters(cube), beta, atol=1e-15)


@pytest.mark.parametrize("bad", [np.zeros((2, 1, 3)), np.full((2, 2, 3), np.nan)])
def test_missing_key_support_is_not_filled(bad):
    """An incomplete finite object cannot be promoted by defaults."""
    with pytest.raises(ValueError):
        M.key_characters(bad)


def test_source_common_shape_and_role_permissions_fail_closed():
    """Do not mix registered addresses or silently allow continuous doses."""
    x = np.zeros((4, 2, 3))
    with pytest.raises(ValueError):
        M.reconstruct_key(x, 1, 1, 0.5, 1)
    with pytest.raises(ValueError):
        M.reconstruct_key(x, 1, 1, 1, 0, common_role=np.zeros((1, 3)))
