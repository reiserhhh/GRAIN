#!/usr/bin/env python3
"""Signed component interactions must reconstruct total covariance, not RMS shares."""

import numpy as np

from grain_research.vector_mechanisms.construction_bridge import covariance_matrix
from scripts.summarize_grain_functional_construction_bridge import component_covariance


def test_sixteen_term_sum_preserves_opposing_channels():
    rng = np.random.default_rng(27)
    parts = {str(k): rng.normal(size=(12, 2, 2, 3)) for k in range(4)}
    parts["1"] = -0.8 * parts["0"]
    total = sum(parts.values())
    account = component_covariance(parts, 0, 0)
    np.testing.assert_allclose(account.sum(), covariance_matrix(total)[0, 0])
    assert account[0, 1] < 0
    np.testing.assert_allclose(
        component_covariance(parts, 1, 2).sum(), covariance_matrix(total)[1, 2]
    )
