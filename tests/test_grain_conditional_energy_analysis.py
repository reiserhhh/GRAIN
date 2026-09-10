#!/usr/bin/env python3
"""Scale-free correlation and author-aware interpretation checks."""

import numpy as np

from scripts.analyze_grain_conditional_energy import author_draws, corr, within


def test_correlation_keeps_tiny_representable_variation():
    x = np.array([1.0, 3.0, -2.0, 4.0])
    y = np.array([1.2, 2.0, -0.3, 4.0])
    np.testing.assert_allclose(corr(x * 1e-30, y * 1e-25), corr(x, y))
    assert np.isnan(corr(np.ones(4), y))


def test_constant_fold_baseline_is_exactly_zero_within_fold():
    groups = np.repeat(np.arange(4), 6)
    x = np.repeat([0.00023, 0.00017, 0.00041, 0.000011], 6)
    np.testing.assert_array_equal(within(x, groups), 0)
    assert np.isnan(corr(within(x, groups), np.arange(24)))


def test_author_bootstrap_keeps_both_pairs():
    ids = np.repeat(["a", "b", "c"], 2)
    for draw in author_draws(ids)[:20]:
        counts = np.bincount(draw, minlength=6).reshape(3, 2)
        np.testing.assert_array_equal(counts[:, 0], counts[:, 1])
