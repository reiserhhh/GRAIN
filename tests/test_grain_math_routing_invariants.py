#!/usr/bin/env python3
"""Exact synthetic consequences and counterexamples for conditional reading laws."""

import numpy as np
import pytest

from grain_research.construction.math_routing_invariants import restrictions


def test_scalar_gate_is_nested_in_all_laws():
    """A common multiplier preserves every required polynomial identity."""
    a = np.random.default_rng(1).normal(size=(6, 6))
    r = restrictions(a, -0.3 * a)
    assert r["scalar_residual_squared_share"] < 1e-28
    for key in ("row_only", "query_only", "row_and_query"):
        assert r[key]["relative_polynomial_norm"] < 1e-14


def test_two_sided_gate_passes_quartics_but_not_simple_gates():
    """Content and target scaling can be separable without a scalar gate."""
    a = np.random.default_rng(2).uniform(0.2, 1, (6, 6))
    b = np.arange(1, 7)[:, None] * a * np.arange(7, 13)[None, :]
    r = restrictions(a, b)
    assert r["row_and_query"]["relative_polynomial_norm"] < 1e-14
    assert r["row_only"]["relative_polynomial_norm"] > 0.1
    assert r["query_only"]["relative_polynomial_norm"] > 0.1


def test_single_crossing_change_breaks_two_sided_law():
    """A condition-specific interaction is detectable without cellwise division."""
    a = np.ones((6, 6))
    b = a.copy()
    b[2, 4] = -1
    r = restrictions(a, b)
    assert r["row_and_query"]["relative_polynomial_norm"] > 0.1
    assert r["row_and_query"]["constraints"] == 225


def test_zero_support_is_not_identification():
    """Vacuous zero polynomial is not evidence that a conditioning mechanism fits."""
    r = restrictions(np.zeros((6, 6)), np.ones((6, 6)))
    assert r["zero_support_caveat"]
    assert r["row_and_query"]["relative_polynomial_norm"] is None
    assert r["scalar_residual_squared_share"] is None


def test_unit_change_does_not_change_normalized_invariants():
    """Common output unit changes preserve the normalized algebraic discrepancy."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(6, 6)), rng.normal(size=(6, 6))
    first, second = restrictions(a, b), restrictions(10 * a, 10 * b)
    for key in ("row_only", "query_only", "row_and_query"):
        assert first[key]["relative_polynomial_norm"] == pytest.approx(
            second[key]["relative_polynomial_norm"]
        )
