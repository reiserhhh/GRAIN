#!/usr/bin/env python3
"""F04 response-level mathematical checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f04 import CONDITIONS, design_matrix, generate_world, information


def test_design_rank_certificates():
    assert information("linear")["rank"] == 2
    assert information("quadratic")["rank"] == 3
    assert information("four_cell")["rank"] == 4


def test_response_worlds_have_separate_truth_and_records():
    rows, truth = generate_world("H", 7, counts={"source": 1, "validation": 1, "application": 1}, events=8)
    assert len(truth) == 3
    assert len(rows) == 24
    assert all("a" not in row and "b" not in row for row in rows)
    assert set(row["z"] for row in rows).issubset(set(CONDITIONS))


def test_repeated_events_do_not_change_design_rank():
    x = design_matrix("linear")
    assert np.linalg.matrix_rank(np.vstack([x, x])) == np.linalg.matrix_rank(x)
