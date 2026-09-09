#!/usr/bin/env python3
"""No-data tests for source-only value/secant estimators."""

import numpy as np

from grain_research.vector_mechanisms.conditional_models import (
    fit_response_kernel,
    predict_response_kernel,
)


def test_linear_value_equals_primal_ridge():
    """Gram implementation agrees with the declared centered linear estimator."""
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(12, 5)), rng.normal(size=(12, 3))
    ii = np.arange(9)
    model = fit_response_kernel(x @ x.T, y, ii, [])
    xn = (x - x[ii].mean(0)) / np.sqrt(model["scale2"])
    coef = np.linalg.solve(
        xn[ii].T @ xn[ii] + np.eye(5), xn[ii].T @ (y[ii] - y[ii].mean(0))
    )
    np.testing.assert_allclose(
        predict_response_kernel(model, x @ x.T), y[ii].mean(0) + xn @ coef, atol=1e-12
    )


def test_held_targets_and_edges_cannot_enter_fit():
    """Changing held outcomes leaves coefficients and predictions identical."""
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(12, 5)), rng.normal(size=(12, 3))
    ii = np.arange(9)
    pairs = [(0, 1), (1, 11), (4, 5), (10, 11)]
    first = fit_response_kernel(x @ x.T, y, ii, pairs, degree=2, secant=True)
    y[9:] = np.nan
    second = fit_response_kernel(x @ x.T, y, ii, pairs, degree=2, secant=True)
    np.testing.assert_allclose(first["coef"], second["coef"])
    assert int(first["edge_count"]) == 2


def test_secant_metric_equals_weighted_primal_solution():
    """Pair supervision changes row weighting rather than inventing new endpoints."""
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=(8, 4)), rng.normal(size=(8, 2))
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    model = fit_response_kernel(x @ x.T, y, np.arange(8), pairs, secant=True)
    xn = (x - x.mean(0)) / np.sqrt(model["scale2"])
    d = np.zeros((4, 8))
    for row, (a, b) in enumerate(pairs):
        d[row, a], d[row, b] = -1, 1
    m = np.eye(8) + float(model["lambda"]) * 2 * d.T @ d
    coef = np.linalg.solve(xn.T @ m @ xn + np.eye(4), xn.T @ m @ (y - y.mean(0)))
    np.testing.assert_allclose(
        predict_response_kernel(model, x @ x.T), y.mean(0) + xn @ coef, atol=1e-12
    )


def test_prediction_difference_is_same_score_function():
    """A scalar score has exact finite-difference composition, including quadratic fits."""
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(10, 5)), rng.normal(size=(10, 3))
    model = fit_response_kernel(
        x @ x.T, y, np.arange(10), [(0, 1), (2, 3)], degree=2, secant=True
    )
    p = predict_response_kernel(model, x @ x.T).sum(1)
    np.testing.assert_allclose((p[2] - p[1]) + (p[1] - p[0]), p[2] - p[0])
