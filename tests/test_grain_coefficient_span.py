#!/usr/bin/env python3
"""A source span limitation cannot be removed by changing only its penalty."""

import numpy as np

from scripts.analyze_grain_coefficient_span import span_account


def test_source_ridge_error_splits_into_span_and_shrinkage():
    x = np.column_stack([np.ones(4), [-1.0, -1.0, 1.0, 1.0]])
    y = np.column_stack([[0.0, 2.0, 2.0, 4.0], [1.0, 1.0, 1.0, 1.0]])
    w = np.full(4, 0.25)
    coef = np.array([[2.0, 1.0], [0.5, 0.0]])
    d = span_account(x, y, w, coef)
    assert d["best_span_error"] > 0
    assert d["ridge_penalty_error"] > 0
    assert abs(d["identity_error"]) < 1e-12


def test_exact_rank_deficient_design_has_zero_span_error():
    x = np.column_stack([np.ones(10), np.arange(10.0), np.arange(10.0)])
    coef = np.array([[2.0, 1.0], [0.5, -0.2], [0.3, 0.2]])
    d = span_account(x, x @ coef, np.full(10, 0.1), coef)
    assert d["source_span_rank"] == 2
    assert d["best_span_error"] < 1e-25
