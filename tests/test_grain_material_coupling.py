#!/usr/bin/env python3
"""Numeric contracts for local co-location and lexical/subword address alignment."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.material_coupling import (
    local_coupling,
    row_correlations,
    span_allocation,
)


def test_weighted_change_is_covariance_not_added_information():
    """Direct arithmetic and the conditional covariance identity must agree."""
    g = np.array([[0, 3], [1, 0], [5, 2]], float)
    r = local_coupling(g, [0, 1, 7])
    assert r["identity_error"] < 1e-12
    np.testing.assert_allclose(r["W"] - r["U"], r["D"])


def test_score_offset_and_sign_do_not_change_salience_weighting():
    """The salience statistic is reference-offset invariant, the signed one is not."""
    g = np.arange(12).reshape(4, 3)
    p = np.array([0, 2, 1, 8])
    a, b = local_coupling(g, p), local_coupling(g, -p + 13)
    np.testing.assert_allclose(a["W"], b["W"])
    np.testing.assert_allclose(a["S"], -b["S"])


def test_constant_function_has_uniform_fallback():
    """No local contrast must not invent evidence selection."""
    r = local_coupling(np.arange(8).reshape(4, 2), np.ones(4))
    np.testing.assert_allclose(r["W"], r["U"])
    np.testing.assert_allclose(r["D"], 0)
    assert r["effective_comments"] == 4


def test_tiny_nonconstant_function_uses_one_threshold():
    """The identity uses the same total-salience fallback as its weights."""
    g = np.array([[0], [1], [3], [8]], float)
    result = local_coupling(g, np.array([0, 1, 2, 7]) * 4e-15)
    assert result["identity_error"] < 1e-12
    assert not np.allclose(result["W"], result["U"])


def test_alignment_is_not_bag_marginals():
    """The same two marginal bags can produce different coupled statistics."""
    g, p = np.array([[0], [1], [4], [9]]), np.array([0, 1, 2, 8])
    a, b = local_coupling(g, p), local_coupling(g, np.roll(p, 1))
    np.testing.assert_allclose(a["U"], b["U"])
    assert not np.allclose(a["W"], b["W"])


def test_effect_allocation_preserves_signed_total_across_token_boundary():
    """A subword token crossing a lexical cut is allocated, not duplicated."""
    a = span_allocation([[0, 2], [2, 6], [6, 9]], [[0, 4], [4, 9]])
    effect = np.array([1, -4, 2])
    np.testing.assert_allclose(a.sum(0), 1)
    np.testing.assert_allclose((a @ effect).sum(), effect.sum())
    assert a[0, 1] == 0.5


def test_refinement_reaggregates_without_new_attribution():
    """Refining display slices preserves the original account."""
    tokens = [[0, 3], [3, 8]]
    coarse = span_allocation(tokens, [[0, 8]])
    fine = span_allocation(tokens, [[0, 2], [2, 5], [5, 8]])
    np.testing.assert_allclose(fine.sum(0), coarse[0])


def test_missing_or_overlapping_slice_domain_rejected():
    """A partial map must not silently drop or double-count evidence."""
    with pytest.raises(ValueError):
        span_allocation([[0, 5]], [[0, 2], [3, 5]])
    with pytest.raises(ValueError):
        span_allocation([[0, 5]], [[0, 3], [2, 5]])
    with pytest.raises(ValueError):
        span_allocation([[0, 5]], [[1, 5]])


def test_bootstrap_row_correlation_matches_scalar():
    """Row-wise computation preserves correlation and leaves constants undefined."""
    a = np.array([[1, 2, 4], [3, 3, 3]], float)
    b = np.array([[4, 2, 1], [1, 2, 3]], float)
    r = row_correlations(a, b)
    assert abs(r[0] - np.corrcoef(a[0], b[0])[0, 1]) < 1e-12
    assert np.isnan(r[1])
