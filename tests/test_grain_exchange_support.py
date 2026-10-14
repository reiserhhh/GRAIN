#!/usr/bin/env python3
"""Natural exchange constraints, distinct from prediction and psychological truth."""

import itertools

import numpy as np
import pytest

from grain_research.construction.math_exchange_support import exchange_parts
from grain_research.construction.math_qualified_target_field import characters


@pytest.mark.parametrize("axis,coordinate", [("e", 0), ("n", 1)])
def test_any_natural_exchange_has_only_odd_action(axis, coordinate):
    """This holds for arbitrary nonlinear readouts, not just linear models."""
    cube = np.random.default_rng(161).normal(size=(2, 2, 2))
    delta = np.flip(cube, axis=coordinate) - cube
    result = exchange_parts(characters(delta), axis)
    np.testing.assert_allclose(result["even"], 0, atol=1e-15)


def test_relation_only_can_change_the_other_behavior_without_an_actor_exchange():
    """At r=en, flipping only r leaves the natural four-record support."""
    action = np.zeros((2, 2, 2))
    for ei, ni, pi in itertools.product(range(2), repeat=3):
        e, n, p = 2 * ei - 1, 2 * ni - 1, 2 * pi - 1
        r = e * n
        natural_readout = r * e * p
        edited_readout = -r * e * p
        action[ei, ni, pi] = edited_readout - natural_readout
    result = exchange_parts(characters(action), "e")
    assert result["even_energy_fraction"] == 1
    assert characters(action)[6] == -2


def test_null_action_is_not_perfect_exchange_evidence():
    """Zero action supplies no informative denominator."""
    assert exchange_parts(np.zeros(8), "e")["even_energy_fraction"] is None


def test_invalid_support_fails_closed():
    """Missing character terms cannot establish group-action compatibility."""
    with pytest.raises(ValueError):
        exchange_parts(np.zeros(7), "e")
    with pytest.raises(ValueError):
        exchange_parts(np.zeros(8), "unknown")
