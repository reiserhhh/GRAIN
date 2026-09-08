#!/usr/bin/env python3
"""Finite witnesses for the conditional anchor identity, bound and author permutation."""

import numpy as np
import pytest

from scripts.analyze_grain_level_configuration import distance_relation


def test_conditional_intercept_can_vary_with_identical_query_gradient():
    z = np.array([0.1, 0.5, -0.4])
    reference = np.array([0.1, 0.2, 0.3])
    gradient = np.array([1, 2, 3.0])
    effect = gradient @ (z - reference)
    levels = np.array([0.1, 0.9]) + effect
    assert levels[1] - levels[0] == pytest.approx(0.8)
    # Shared derivatives determine change, not the conditional intercept.
    assert levels - [0.1, 0.9] == pytest.approx([effect, effect])


def test_quadratic_function_attains_reference_transport_bound():
    d = np.array([0.3, -0.4])
    lipschitz, eta = 2.0, 0.2
    gradient_mismatch = eta * d / np.linalg.norm(d)
    measured_error = gradient_mismatch @ d + 0.5 * lipschitz * (d @ d)
    bound = eta * np.linalg.norm(d) + 0.5 * lipschitz * np.linalg.norm(d) ** 2
    assert measured_error == pytest.approx(bound)


def test_distance_comparison_counts_authors_not_independent_pairs():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(8, 4))
    result = distance_relation(x, x, draws=50)
    assert result["authors"] == 8
    assert result["pairs"] == 28
    assert result["distance_r"] == pytest.approx(1)
    assert result["two_sided_author_permutation_p"] == pytest.approx(1 / 51)
