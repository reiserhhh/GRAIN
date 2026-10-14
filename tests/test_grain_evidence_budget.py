#!/usr/bin/env python3
"""Known-model and finite-sampling tests, no private data."""

import numpy as np
import pytest

from grain_research.evidence_budget import (
    beta_coverage,
    finite_coverage,
    fit_beta_binomial,
)


def test_finite_pool_extremes():
    assert np.array_equal(finite_coverage([0, 1, 36], 36, 36), [0, 1, 1])
    assert np.allclose(finite_coverage([0, 1, 36], 36, 1), [0, 1 / 36, 1])
    assert np.array_equal(finite_coverage([0, 1, 36], 36, 0), [0, 0, 0])


def test_heterogeneity_reduces_population_coverage():
    a, b = 0.3, 15.0
    for budget in (1, 4, 16, 36):
        assert beta_coverage(budget, a, b) <= 1 - (1 - a / (a + b)) ** budget + 1e-12


def test_mixture_recovers_coverage():
    rng = np.random.default_rng(9)
    counts = rng.binomial(36, rng.beta(0.4, 10, size=5000))
    fit = fit_beta_binomial(counts, 36)
    assert beta_coverage(24, fit["alpha"], fit["beta"]) == pytest.approx(
        beta_coverage(24, 0.4, 10), abs=0.02
    )


def test_invalid_sampling():
    with pytest.raises(ValueError):
        finite_coverage([1], 36, 40)
    with pytest.raises(ValueError):
        fit_beta_binomial([1.3, 2], 36)
