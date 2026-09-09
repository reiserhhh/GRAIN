#!/usr/bin/env python3
"""Finite clock counterexamples, not a new event scorer or human experiment."""

from itertools import permutations

import numpy as np


def chronological_change(events):
    """Canonicalize a synthetic event list with unique observed times."""
    ordered = sorted(events)
    return ordered[-1][1] - ordered[0][1]


def test_display_invariance_can_preserve_actual_time_response():
    events = [(0, 0.0), (1, 1.0), (2, 3.0)]
    for displayed in permutations(events):
        assert chronological_change(displayed) == 3
    actual_reversal = [(0, 3.0), (1, 1.0), (2, 0.0)]
    assert chronological_change(actual_reversal) == -3
    assert np.mean([v for _, v in events]) == np.mean([v for _, v in actual_reversal])


def test_removing_clock_metadata_makes_two_histories_indistinguishable():
    displayed_reversal = [(2, "c"), (1, "b"), (0, "a")]
    actual_reversal = [(0, "c"), (1, "b"), (2, "a")]
    strip = lambda rows: [value for _, value in rows]
    assert strip(displayed_reversal) == strip(actual_reversal)
    assert sorted(displayed_reversal) != sorted(actual_reversal)


def test_symmetric_display_average_removes_a_real_directional_statistic():
    original = [(0, 0.0), (1, 1.0), (2, 3.0)]
    changed = [(0, 3.0), (1, 1.0), (2, 0.0)]
    assert (chronological_change(original) + chronological_change(changed)) / 2 == 0
    assert chronological_change(original) != 0
