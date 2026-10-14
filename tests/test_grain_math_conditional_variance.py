#!/usr/bin/env python3
"""Finite product-space tests; no corpus or model dependencies."""

import numpy as np
import pytest

from grain_research.construction.math_conditional_variance import product_components


def test_additive_product_has_no_spurious_interaction():
    x = np.array([-1, 1])[:, None, None]
    y = np.array([-1, 0, 1])[None, :, None]
    z = np.array([-1, 1])[None, None, :]
    result = product_components(2 + x + y + z)
    assert result["mean"] == 2
    assert result["energies"][7] < 1e-25
    assert result["max_orthogonality_error"] < 1e-12
    assert result["variance_closure_error"] < 1e-12


def test_zero_marginal_effect_can_hide_pure_interaction():
    x = np.array([-1, 1])[:, None]
    y = np.array([-1, 1])[None, :]
    result = product_components(x * y)
    assert result["energies"][1] == result["energies"][2] == 0
    assert result["energies"][3] == 1
    assert result["max_reconstruction_error"] == 0


def test_missing_or_degenerate_product_is_not_silently_imputed():
    for bad in (np.array([1, np.nan]), np.ones((1, 2)), np.asarray(3)):
        with pytest.raises(ValueError):
            product_components(bad)
