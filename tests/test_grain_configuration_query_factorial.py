#!/usr/bin/env python3
"""Finite nonlinear witnesses distinguish main effects from shared interaction."""

import numpy as np
import pytest

from scripts.run_grain_configuration_query_factorial import factorial_cells


def test_exact_decomposition_and_no_double_counted_configuration():
    e, x = np.array([1.0, 2.0]), np.array([3.0, -1.0])
    b = np.array([0.3, -0.4])
    f11, f01, f10, f00 = b + e + x + e * x, b + x, b + e, b
    parts = factorial_cells(f11, f01, f10, f00)
    assert parts["E_main"] == pytest.approx(e)
    assert parts["X_main"] == pytest.approx(x)
    assert parts["interaction"] == pytest.approx(e * x)
    assert sum(parts.values()) == pytest.approx(f11)
    assert parts["E_main"] + parts["interaction"] == pytest.approx(f11 - f01)


def test_additive_function_has_zero_interaction():
    parts = factorial_cells([6.0, 10.0], [4.0, 7.0], [3.0, 5.0], [1.0, 2.0])
    assert parts["interaction"] == pytest.approx([0, 0])
