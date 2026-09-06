#!/usr/bin/env python3
"""Synthetic exact-overlap and conditional-association controls."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.domain_overlap import (
    ngram_coverage,
    residual_correlation,
    subset_effect,
)


def test_all_covered_tokens_and_partial_repetitions():
    np.testing.assert_array_equal(
        ngram_coverage([1, 2, 3, 4, 5, 8], [0, 2, 3, 4, 5, 9], 4), [0, 1, 1, 1, 1, 0]
    )
    np.testing.assert_array_equal(ngram_coverage([1, 2, 3], [1, 2], 4), [0, 0, 0])
    np.testing.assert_array_equal(ngram_coverage([7, 7, 8], [7], 1), [1, 1, 0])
    with pytest.raises(ValueError):
        ngram_coverage([1], [1], 0)


def test_case_minimum_not_imputed_or_counted_as_tokens():
    assert np.isnan(subset_effect(np.ones(8), [1, 1, 1, 0, 0, 0, 0, 0]))
    assert subset_effect(np.arange(8), [1, 1, 1, 1, 0, 0, 0, 0]) == 1.5


def test_nuisance_only_is_undefined_and_residual_recoverable():
    rng = np.random.default_rng(20)
    z = np.column_stack((np.ones(100), rng.normal(size=(100, 3))))
    x = z @ [1, 2, 3, 4]
    assert np.isnan(residual_correlation(x, x, z)[0])
    residual = rng.normal(size=100)
    r, rank, dof = residual_correlation(x + residual, 2 * x + residual, z)
    assert r == pytest.approx(1)
    assert rank == 4 and dof == 96
