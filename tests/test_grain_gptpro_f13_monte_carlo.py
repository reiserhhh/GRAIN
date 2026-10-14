#!/usr/bin/env python3
"""Exact enumeration, not fitted simulation, verifies nested covariance bias."""

import itertools

import numpy as np

from grain_research.gptpro_foundation_v3.f13_monte_carlo import (
    inner_sampling_correction,
)


def test_exact_two_source_two_draw_enumeration():
    results = []
    # Fixed source means 0,2 and independent within-source variances 1,4.
    for signs in itertools.product([-1.0, 1.0], repeat=4):
        samples = np.array(signs).reshape(2, 2) * np.array([[1.0], [2.0]]) + np.array(
            [[0.0], [2.0]]
        )
        personal = samples.var(axis=1).mean()
        source = samples.mean(axis=1).var()
        corrected = inner_sampling_correction(personal, source, 2, 2)
        results.append(
            [
                personal,
                source,
                corrected["personal_corrected"],
                corrected["source_corrected"],
                personal + source + corrected["total_correction"],
            ]
        )
    np.testing.assert_allclose(
        np.mean(results, axis=0), [1.25, 1.625, 2.5, 1.0, 3.5], atol=1e-12
    )
    assert min(r[3] for r in results) < 0  # A finite unbiased estimate need not be PSD.
