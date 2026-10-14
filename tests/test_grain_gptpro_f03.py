#!/usr/bin/env python3
"""F03 opportunity/selection/response checks."""

import pytest

from grain_research.gptpro_foundation_v3.f03 import (
    WORLD_VALUES,
    cell_probability,
    generate_records,
    known_opportunity_bounds,
    response_bound,
    response_probability,
    selection_probability,
)


def test_worlds_and_probabilities_are_bounded():
    assert set(WORLD_VALUES) == {"W3-A", "W3-B", "W3-C", "W3-D"}
    for r in (0.2, 0.5, 0.8):
        for x in (0, 1):
            assert 0.0 < response_probability(r, x) < 1.0
    for gamma in (0.0, 1.0):
        assert 0.0 < selection_probability(0.0, 0.8, gamma, 1, 1, 1) < 1.0
    cell = cell_probability(0.5, 0.4, 0.0, 0.8, 1.0, 1, 0, 1)
    assert 0.0 < cell["p_published_y"] < cell["p_opportunity_and_y"]


def test_unknown_denominator_bounds_are_wider():
    assert response_bound(0.2, 0.4) == pytest.approx((0.2, 0.8))
    assert response_bound(0.2, 0.4, 0.8) == pytest.approx((0.25, 0.75))


def test_fixed_design_has_one_observation_per_event_and_truth_is_separate():
    records, truth = generate_records("W3-C", "fixed", 7, events=8, counts={"source": 1, "validation": 1, "application": 1})
    assert len(truth) == 3
    assert all(row["selected"] == 1 and row["observation"] in (0, 1) for row in records)
    assert all("r" not in row and "pi" not in row for row in records)
    lower, upper, f, pi = known_opportunity_bounds(records, -1)
    assert f == pytest.approx(1.0)
    assert pi == pytest.approx(1.0)
    assert 0.0 <= lower <= upper <= 1.0
