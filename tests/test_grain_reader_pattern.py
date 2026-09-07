#!/usr/bin/env python3
"""Synthetic score identities, coordinate equivariance and nonlinear interaction."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.reader_pattern import (
    effect_decomposition,
    reader_patterns,
    split_reader_component,
)


def test_forward_pattern_equals_least_squares_and_score_is_preserved():
    rng = np.random.default_rng(18)
    x, beta, d = rng.normal(size=(20, 40)), rng.normal(size=40), rng.normal(size=40)
    patterns, _ = reader_patterns(x, beta)
    xc = x - x.mean(0)
    score = xc @ beta
    fitted = np.linalg.lstsq(score[:, None], xc, rcond=None)[0][0]
    np.testing.assert_allclose(patterns["covariance"], fitted, atol=1e-12)
    for pattern in patterns.values():
        selected, null = split_reader_component(d, beta, pattern)
        np.testing.assert_allclose(selected + null, d, atol=1e-12)
        assert beta @ null == pytest.approx(0, abs=1e-12)
        assert beta @ selected == pytest.approx(beta @ d)


def test_covariance_pattern_is_equivariant_under_nonorthogonal_coordinates():
    rng = np.random.default_rng(19)
    x, beta = rng.normal(size=(30, 5)), rng.normal(size=5)
    transform = np.diag([1, 2, 3, 4, 5.0])
    transform[0, 2] = 0.7
    old, _ = reader_patterns(x, beta)
    new, _ = reader_patterns(x @ transform.T, np.linalg.solve(transform.T, beta))
    np.testing.assert_allclose(
        new["covariance"], transform @ old["covariance"], atol=1e-12
    )
    assert not np.allclose(new["euclidean"], transform @ old["euclidean"])
    with pytest.raises(ValueError, match="positive"):
        reader_patterns(np.ones((5, 5)), beta)


def test_nonlinear_rectangle_keeps_interaction():
    h, selected, null = np.array([2.0, 3.0]), np.array([1.0, 0.0]), np.array([0.0, 2.0])
    f = lambda x: x[0] * x[1]
    result = effect_decomposition(
        f(h), f(h - selected - null), f(h - selected), f(h - null)
    )
    assert result["interaction"] == pytest.approx(-2.0)


def test_covariance_pattern_minimizes_supported_mahalanobis_displacement():
    rng = np.random.default_rng(20)
    x, beta = rng.normal(size=(8, 12)), rng.normal(size=12)
    patterns, _ = reader_patterns(x, beta)
    centered = x - x.mean(0)
    sigma = centered.T @ centered / (len(x) - 1)
    precision = np.linalg.pinv(sigma)
    selected = patterns["covariance"] * 0.7
    direction = centered.T @ rng.normal(size=len(x))
    null = direction - patterns["covariance"] * (beta @ direction)
    assert beta @ null == pytest.approx(0, abs=1e-12)
    assert selected @ precision @ null == pytest.approx(0, abs=1e-12)
    for amplitude in (-2.0, -0.1, 0.1, 2.0):
        alternative = selected + amplitude * null
        assert alternative @ precision @ alternative >= selected @ precision @ selected
