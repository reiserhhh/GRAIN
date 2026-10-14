#!/usr/bin/env python3
"""Exact support worlds for the causal-prefix interaction law."""

import itertools

import numpy as np
import pytest

from grain_research.construction.math_event_field import characters
from scripts.inspect_grain_event_field_prefix_support import allowed, availability


def test_prefix_availability_uses_actual_flips_not_word_labels():
    cells = list(itertools.product((-1, 1), repeat=3))
    assert availability([(x,) for x, y, g in cells]) == (True, False, False)
    assert availability([(y, g) for x, y, g in cells]) == (False, True, True)


def test_missing_factor_forces_fourier_coefficients_to_zero():
    cells = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    field = cells[:, 1] + 3 * cells[:, 2] + cells[:, 1] * cells[:, 2]
    coefficients = characters() @ field / 8
    for mask in range(1, 8):
        if not allowed(mask, (False, True, True)):
            assert coefficients[mask - 1] == 0


def test_supported_interactions_can_move_to_disjoint_event_slots():
    forward = ((True, False, False), (True, True, True), (True, True, True))
    reverse = ((True, True, True), (False, True, True), (False, False, True))
    for mask in (3, 5, 7):
        a = np.array([allowed(mask, s) for s in forward], float)
        b = np.array([allowed(mask, s) for s in reverse], float)
        assert np.dot(a, b) == 0
        assert a.sum() > 0 and b.sum() > 0


def test_incomplete_prefix_support_rejected():
    with pytest.raises(ValueError):
        availability([()] * 7)
