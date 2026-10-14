#!/usr/bin/env python3
"""Synthetic tests for complete observation programs, not human validation."""

import numpy as np
import pytest

from grain_research.construction import math_observation_program as M


def test_exact_characters_and_program_interaction():
    """Eight characters preserve the three-way contrast and every corner."""
    rng = np.random.default_rng(136)
    x = rng.normal(size=(2, 2, 2, 2, 7, 4))
    g = M.characters(x)
    np.testing.assert_allclose(M.reconstruct(g), x, atol=2e-15)
    e = M.effects(x)
    np.testing.assert_allclose(e[1, 3] - e[0, 3], 8 * g[7], atol=2e-15)


def test_zero_value_addition_not_information_monotonicity():
    """More accessible carriers can create an interaction by normalization alone."""
    x = M.zero_value_carrier_example()
    e = M.effects(x)
    assert e[0, 3] == 0
    assert e[1, 3] == -0.5
    assert M.characters(x)[7] == -0.5 / 8


def test_target_person_unit_axes_not_pooled():
    """Actor, nonactor and units remain separately identifiable in the export."""
    x = np.zeros((2, 2, 2, 2, 7, 4))
    x[1, 1, 1, 1, 6, 2] = 8
    e = M.effects(x)
    assert e.shape == (2, 4, 2, 7, 4)
    assert e[1, 3, 1, 6, 2] == 8
    assert np.count_nonzero(e[0]) == 0
    assert np.count_nonzero(e[..., 0, :, :]) == 0


def test_nonfinite_or_incomplete_support_fails():
    """A missing finite corner cannot silently become zero or a reduced claim."""
    with pytest.raises(ValueError):
        M.effects(np.zeros((2, 2, 1)))
    x = np.zeros((2, 2, 2))
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        M.characters(x)


def test_signed_and_zero_field_comparison():
    """Preserve cancellation and undefined ratios rather than selecting a winner."""
    a = np.array([1.0, -2.0, 3.0])
    d = M.field_comparison(a, -2 * a)
    assert d["cosine"] == -1
    assert d["full_signed_projection_on_H"] == -2
    assert d["relative_difference_to_H"] == 3
    d = M.field_comparison(np.zeros(3), a)
    assert d["cosine"] is None
    assert d["full_signed_projection_on_H"] is None


def test_fixed_complete_execution_support():
    """Control selection is registered by address, not desirable output."""
    from scripts import run_grain_observation_program_hm136 as R

    spec = R.H.load(R.SPEC)
    design = R.fixed_design([{}, {}], spec)
    assert len(design) == 108
    full = [d for d in design if d["route"] == 7 and not d["noop"]]
    assert len(full) == len({(d["source"], d["row"]) for d in full}) == 96
    assert sum(d["route"] == 1 for d in design) == 8
    assert sum(d["noop"] for d in design) == 4


def test_native_query_slot_mapping():
    """The factual A/B slots cannot be mixed into trait channels."""
    from scripts import run_grain_observation_program_hm136 as R

    x = np.arange(56).reshape(14, 4)
    y = R.person_units({"readings": x})
    np.testing.assert_array_equal(y[0, :6], x[:6])
    np.testing.assert_array_equal(y[1, :6], x[6:12])
    np.testing.assert_array_equal(y[0, 6], x[12])
    np.testing.assert_array_equal(y[1, 6], x[13])
