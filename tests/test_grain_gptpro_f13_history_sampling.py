#!/usr/bin/env python3
"""Exact path enumeration tests for conditional-history sampling."""

import itertools

import numpy as np

from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory
from grain_research.gptpro_foundation_v3.f13_history_sampling import (
    path_counts,
    sample_histories,
)


def test_full_history_law_and_online_difference():
    """FFBS matches enumeration but must not be relabeled online-count uncertainty."""
    p = np.array([0.2, 0.8, 0.3, 0.9, 0.1, 0.8])
    q = np.array([1, 0.8, 0.5, 0.9, 0.7, 1])
    transition = np.array([[0.1, 0.8], [0.7, 0.2]])
    initial = np.array([[0.1, 0.2], [0.3, 0.4]])
    paths = np.array(list(itertools.product((0, 1), repeat=6)))
    weights = []
    for path in paths:
        value = initial[path[0], path[1]]
        for t, y in enumerate(path):
            value *= q[t] * (p[t] if y else 1 - p[t]) + (1 - q[t]) / 2
            if t >= 2:
                prob = transition[path[t - 2], path[t - 1]]
                value *= prob if y else 1 - prob
        weights.append(value)
    weights = np.array(weights) / sum(weights)
    sampled = sample_histories(
        p, q, transition, initial, 100000, np.random.default_rng(1313)
    )
    sample_counts = path_counts(sampled, q).mean(axis=0)
    exact_counts = np.einsum("b,bij->ij", weights, path_counts(paths, q))
    np.testing.assert_allclose(sample_counts, exact_counts, atol=0.01)
    np.testing.assert_allclose(sampled.mean(axis=0), weights @ paths, atol=0.01)
    online = QualifiedHistory({"b1": {"source_p": transition.ravel()}}, initial)
    for a, b in zip(p, q, strict=True):
        online.observe(a, b)
    assert np.max(abs(online.counts - exact_counts)) > 0.01


def test_hard_complete_path_is_unique():
    """Hard fully observed data have no latent history ambiguity."""
    p = [0, 1, 1, 0]
    result = sample_histories(
        p,
        [1] * 4,
        np.full((2, 2), 0.5),
        np.full((2, 2), 0.25),
        200,
        np.random.default_rng(1),
        complete=True,
    )
    np.testing.assert_array_equal(result, np.tile(p, (200, 1)))
