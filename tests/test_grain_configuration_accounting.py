#!/usr/bin/env python3
"""Retained-path algebra and visible cancellation, including outside-body tokens."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.configuration_accounting import (
    covariance_accounting,
    retained_paths,
)
from grain_research.vector_mechanisms.query_tilt import pairing_component


def test_remaining_vector_has_exact_weighted_mean_expression():
    rng = np.random.default_rng(26)
    p = rng.dirichlet(np.ones(9))
    values = rng.normal(size=(9, 7))
    projected = rng.normal(size=7)
    groups = [[0, 1], [4, 5, 6]]
    pairing = pairing_component(p, values, groups)
    outside = [2, 3, 7, 8]
    expected = p[outside] @ values[outside] - projected
    for g in groups:
        expected += p[g].sum() * values[g].mean(0)
    np.testing.assert_allclose(p @ values - projected - pairing, expected, atol=1e-14)


def test_vector_and_functional_addition_and_coordinate_invariance():
    rng = np.random.default_rng(226)
    c, p = rng.normal(size=(2, 20, 3, 5))
    gradient = rng.normal(size=5)
    remainder, scores = retained_paths(c, p, gradient)
    np.testing.assert_allclose(remainder + p, c, atol=1e-14)
    np.testing.assert_allclose(
        scores["pairing"] + scores["remaining"], scores["configuration"], atol=1e-14
    )
    transform = rng.normal(size=(5, 5)) + 5 * np.eye(5)
    _r, transformed = retained_paths(
        c @ transform.T, p @ transform.T, np.linalg.solve(transform.T, gradient)
    )
    for name in scores:
        np.testing.assert_allclose(scores[name], transformed[name], atol=1e-13)


def test_covariance_records_opposing_signal_without_removing_it():
    y = np.arange(15, dtype=float)
    pair, remainder = 2 * y, -1.5 * y
    info = covariance_accounting(y, pair + remainder, pair, remainder)
    assert info["source_covariances"]["pairing"] > 0
    assert info["source_covariances"]["remaining"] < 0
    assert abs(info["source_covariance_identity_error"]) < 1e-12
    assert abs(info["variance_identity_error"]) < 1e-12
    with pytest.raises(ValueError, match="do not add"):
        covariance_accounting(y, pair, pair, remainder)
