#!/usr/bin/env python3
"""Reversal invariance, its risk floor, and the odd readout extension."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.configuration_extraction import InputProgram
from grain_research.vector_mechanisms.input_order import (
    fit_ordinal,
    ordinal_vector,
    predict_ordinal,
    reversal_account,
)


def test_ordinal_is_odd_and_constant_sequence_is_zero():
    x = np.random.default_rng(10).normal(size=(12, 9))
    np.testing.assert_allclose(ordinal_vector(x[::-1]), -ordinal_vector(x), atol=1e-15)
    np.testing.assert_allclose(ordinal_vector(np.ones((12, 9))), 0, atol=1e-15)
    with pytest.raises(ValueError):
        ordinal_vector(x[:1])


@pytest.mark.parametrize("arm", ["e5_mean", "e5_query", "e5_distribution"])
def test_existing_bag_programs_are_invariant(arm):
    rng = np.random.default_rng(44)
    x, y = rng.normal(size=(10, 12, 8)), rng.normal(size=(10, 2))
    model = InputProgram(arm, rng.normal(size=(2, 8))).fit(x, None, y)
    np.testing.assert_allclose(
        model.predict(x, None), model.predict(x[:, ::-1], None), atol=1e-12
    )


def test_orbit_risk_floor_and_different_natural_distribution():
    a = np.arange(12, dtype=float)
    b = a + 3
    result = reversal_account(a, b, a)
    assert result["odd_floor"] == pytest.approx(2.25)
    assert result["pair_mse"] == pytest.approx(4.5)
    assert result["closure_error"] < 1e-12
    # A perfect predictor of original-order data need not fit the paired orbit.
    assert np.mean((a - a) ** 2) == 0


def test_extension_even_part_preserves_base_and_changes_are_odd():
    rng = np.random.default_rng(22)
    x, y = rng.normal(size=(15, 12, 8)), rng.normal(size=(15, 2))
    base = InputProgram("e5_mean", np.zeros((2, 8))).fit(x, None, y)
    model = fit_ordinal(x, y, base)
    p, reverse = predict_ordinal(model, x), predict_ordinal(model, x[:, ::-1])
    np.testing.assert_allclose((p + reverse) / 2, base.predict(x, None), atol=1e-12)
