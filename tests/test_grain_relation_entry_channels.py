#!/usr/bin/env python3
"""Synthetic entry-channel identifiability tests without model calls."""

import numpy as np
import pytest

from grain_research.construction.math_carrier_edit_transport import edit_increment
from grain_research.construction.math_relation_entry_channels import (
    action_description,
    entry_components,
    relation_increment,
)


@pytest.mark.parametrize("e,n", [(-1, -1), (-1, 1), (1, -1), (1, 1)])
def test_relation_is_full_minus_marginal_for_both_edits(e, n):
    """Both edits have the same algebraic mixed term, not the same support."""
    x = np.random.default_rng(158).normal(size=(2, 2, 3, 2))
    for axis in (0, 1):
        expected = edit_increment(x, e, n, axis) - edit_increment(
            x, e, n, axis, marginal=True
        )
        np.testing.assert_allclose(relation_increment(x, e, n), expected, atol=1e-14)


def test_same_joint_different_entry_mechanisms():
    """Identical full observations do not identify K, V or joint interaction."""
    base = np.array([0.4, 0.7])
    delta = np.array([1.0, -2.0])
    key = entry_components(base, base + delta, base, base + delta)
    value = entry_components(base, base, base + delta, base + delta)
    interaction = entry_components(base, base, base, base + delta)
    for result in (key, value, interaction):
        np.testing.assert_allclose(result["joint_entry"], delta)
    np.testing.assert_allclose(key["key_entry"], delta)
    np.testing.assert_allclose(value["value_entry"], delta)
    np.testing.assert_allclose(interaction["recursive_interaction"], delta)


def test_cancellation_is_retained():
    """Large opposing components are not renamed zero signal or noise."""
    result = entry_components([0.0], [2.0], [-1.0], [0.5])
    assert result["recursive_interaction"][0] == -0.5
    np.testing.assert_allclose(
        result["joint_entry"],
        result["key_entry"] + result["value_entry"] + result["recursive_interaction"],
    )
    assert (
        action_description(np.array([2.0]), np.array([0.5]))["signed_projection"] == 4
    )


def test_null_reference_is_not_a_perfect_result():
    """A reference with zero norm supplies no projection denominator."""
    result = action_description(np.array([1.0]), np.array([0.0]))
    assert result["signed_projection"] is None
    assert result["relative_difference"] is None


def test_invalid_input_fails_closed():
    """Incomplete cubes or malformed readouts cannot silently score zero."""
    with pytest.raises(ValueError):
        relation_increment(np.zeros((3, 2)), 1, 1)
    with pytest.raises(ValueError):
        relation_increment(np.zeros((2, 2)), 0, 1)
    with pytest.raises(ValueError):
        entry_components([0], [1, 2], [1], [2])
    with pytest.raises(ValueError):
        action_description(np.array([]), np.array([]))
