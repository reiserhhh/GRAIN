#!/usr/bin/env python3
"""Synthetic tests of native access isolation and lossless observations."""

import numpy as np
import pytest

from grain_research.construction import math_native_access as M


def row():
    """Small causally ordered system/history/relay/current support."""
    return {
        "input_ids": list(range(8)),
        "history": [2, 3, 4],
        "current": [7],
        "padding": [4],
    }


def test_partition_separates_system_from_relay():
    assert M.partition(row()) == {"S": [0, 1], "H": [2, 3, 4], "B": [5, 6], "C": [7]}


def test_partition_rejects_overlap_and_padding_outside_history():
    with pytest.raises(ValueError):
        M.partition({**row(), "current": [3, 7]})
    with pytest.raises(ValueError):
        M.partition({**row(), "padding": [5]})


def test_all_suffix_cuts_leave_body_and_system_intact():
    base = M.permissions(row(), 3, 12, 7)
    off = M.permissions(row(), 3, 12, 0)
    np.testing.assert_array_equal(base[:8], off[:8])
    assert off[8:11, :2].all()
    assert not off[8:, 2:8].any()
    assert not off[:, 4].any()
    assert not off[:, 11].any()
    assert off[9, 8]


@pytest.mark.parametrize("route", range(8))
def test_each_access_bit_is_exact(route):
    mask = M.permissions(row(), 3, 11, route)
    for bit, key in enumerate((2, 5, 7)):
        assert mask[8:11, key].all() == bool(route & (1 << bit))


def test_endpoint_only_cut_leaks_through_early_suffix():
    mask = M.permissions(row(), 3, 11, 7)
    mask[10, 2:8] = False
    # Two attention blocks suffice for H -> early P -> endpoint.
    assert mask[8, 2] and mask[10, 8]
    complete = M.permissions(row(), 3, 11, 0)
    assert not complete[8:, 2].any()


def test_mobius_retains_compensation_and_reconstructs_every_field():
    rng = np.random.default_rng(31)
    fields = rng.normal(size=(8, 2, 6, 4))
    parts = M.mobius(fields)
    np.testing.assert_allclose(M.reconstruct(parts), fields, atol=2e-15)
    np.testing.assert_allclose(parts.sum(axis=0), fields[7], atol=2e-15)
    with pytest.raises(ValueError):
        M.mobius(fields[:7])


def test_qualified_character_and_positive_author_integral():
    signs = [-1, 1]
    corners = np.array(
        [
            [[2 + b + 3 * b * o + 5 * b * o * p for p in signs] for o in signs]
            for b in signs
        ]
    )
    beta = M.characters(corners)
    np.testing.assert_allclose(beta, [2, 1, 0, 3, 0, 0, 0, 5])
    weights = (1 + 0.4 * np.outer(signs, signs)) / 4
    delta = np.sum((weights - (0.5 - weights)) * corners[:, :, 1])
    assert delta == pytest.approx(2 * 0.4 * (beta[3] + beta[7]))


def test_native_units_preserve_logit_shift_and_candidate_mass():
    z = np.arange(10, dtype=float) / 10
    a = M.units(z, 3)
    b = M.units(z + 100, 103)
    np.testing.assert_allclose(a, b, atol=2e-14)
    assert 0 < a[0] < 1
