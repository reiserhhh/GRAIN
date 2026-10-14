#!/usr/bin/env python3
"""Mechanism worlds distinguish event identity, presentation and null fields."""

import itertools

import numpy as np
import pytest

from scripts.diagnose_grain_event_field_cross_order import dot_parts, pairing, summarize


def test_event_identity_survives_when_terminal_position_changes():
    events = np.eye(3)
    assert pairing(events, events)["cosine"] == 1
    assert pairing(events, events)["reversed_slot_cosine"] == pytest.approx(1 / 3)
    assert pairing(events[0], events[-1])["cosine"] == 0


def test_matching_orders_can_hide_an_order_sensitive_field():
    field = np.stack((np.array([1.0, 0]), np.array([0.0, 1])))
    assert pairing(field[0], field[0])["cosine"] == 1
    assert pairing(field[0], field[1])["cosine"] == 0


def test_all_four_dot_products_keep_even_odd_cross_terms():
    rng = np.random.default_rng(72)
    s, n = rng.normal(size=(2, 3, 7)), rng.normal(size=(2, 3, 7))
    p = dot_parts(s, n)
    assert p["even_odd"] != 0 and p["odd_even"] != 0
    for i, j in itertools.product(range(2), repeat=2):
        a, b = 1 - 2 * i, 1 - 2 * j
        actual = (
            p["even_even"]
            + b * p["even_odd"]
            + a * p["odd_even"]
            + a * b * p["odd_odd"]
        )
        np.testing.assert_allclose(actual, np.vdot(s[i], n[j]))


def test_zero_direction_remains_unobserved_not_perfect():
    result = pairing(np.zeros((3, 5)), np.ones((3, 5)))
    assert all(value is None for value in result.values())


def test_vector_pool_does_not_pretend_to_have_event_slots():
    result = pairing(np.ones(5), np.ones(5))
    assert result["cosine"] == pytest.approx(1)
    assert result["cyclic_wrong_event_cosine"] is None
    assert result["reversed_slot_cosine"] is None


def test_empty_summary_does_not_convert_unobserved_to_zero():
    summary = summarize([], [])
    assert len(summary) == 30
    assert all(row["cosine_defined"] == 0 for row in summary)
    assert all(row["cosine_mean"] is None for row in summary)


@pytest.mark.parametrize("bad", [np.ones((1, 3)), np.full((2, 3), np.nan)])
def test_incomplete_nonfinite_fields_rejected(bad):
    with pytest.raises(ValueError):
        dot_parts(bad, bad)
