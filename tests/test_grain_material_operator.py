#!/usr/bin/env python3
"""Exact compiled-query operator identities, not claims about native attention."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.material_geometry import cross_covariance
from grain_research.vector_mechanisms.material_operator import (
    build_operator,
    read_operator,
)


def test_one_operator_reads_all_queries_with_arbitrary_intercepts():
    """The same local operator equals direct covariance for several queries."""
    rng = np.random.default_rng(33)
    g, h = rng.normal(size=(12, 4)), rng.normal(size=(12, 7))
    w, intercept = rng.normal(size=(7, 5)), rng.normal(size=5)
    expected = cross_covariance(g, h @ w + intercept)
    np.testing.assert_allclose(
        read_operator(build_operator(g, h), w), expected, atol=1e-12
    )


def test_invertible_coordinate_change_preserves_functional_readout():
    """Embedding coordinates and query covectors transform in opposite ways."""
    rng = np.random.default_rng(42)
    g, h, w = (
        rng.normal(size=(12, 4)),
        rng.normal(size=(12, 3)),
        rng.normal(size=(3, 2)),
    )
    t = np.array([[2, 1, 0], [0, 3, 2], [1, 0, 4]], float)
    c = build_operator(g, h)
    changed = build_operator(g + 9, h @ t + 3)
    np.testing.assert_allclose(changed, c @ t, atol=1e-12)
    np.testing.assert_allclose(
        read_operator(changed, np.linalg.solve(t, w)), read_operator(c, w), atol=1e-12
    )


def test_matched_reordering_preserves_but_unmatched_reordering_changes_operator():
    """The operator stores pairing but not the order of matched comment pairs."""
    g = np.array([[0, 1], [1, 3], [4, 2]], float)
    h = np.array([[0, 1, 2], [3, 2, 1], [5, 2, 7]], float)
    c = build_operator(g, h)
    np.testing.assert_allclose(build_operator(g[::-1], h[::-1]), c)
    assert not np.allclose(build_operator(g, h[::-1]), c)


def test_dimension_mismatch_rejected():
    """Do not silently reuse weights from a different embedding space."""
    with pytest.raises(ValueError):
        read_operator(np.zeros((4, 8)), np.zeros(7))
