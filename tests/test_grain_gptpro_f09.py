#!/usr/bin/env python3
"""F09 rival process checks."""

from grain_research.gptpro_foundation_v3.f09 import family_law, generate_records, logistic


def test_logistic_and_family_law_are_valid():
    assert logistic(0.0) == 0.5
    for family in ("M1", "M2", "M3", "M4", "belief", "audience"):
        assert 0.0 < family_law((-1.0, 1.0, -1.0), family, 1.0) < 1.0


def test_even_odd_records_and_truth_are_separate():
    rows, truth = generate_records(7, counts={"source": 1, "validation": 1, "application": 1}, events=24)
    assert len(rows) == 72
    assert len(truth) == 3
    assert all("family" not in row and "beta" not in row for row in rows)
