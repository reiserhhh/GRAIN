#!/usr/bin/env python3
"""Check batched source fits against full observation-space conditioning."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f13_history_contract import conditional_moments
from grain_research.gptpro_foundation_v3.f13_masked_conditioning import (
    condition_source_fits,
)


def test_masked_ar_source_batch():
    """Nonconsecutive times retain rho raised to the original gap."""
    times = np.array([0, 1, 4, 9, 10])
    kernel = 0.6 ** abs(times[:, None] - times)
    x = np.column_stack([np.ones(5), [-1, 1, -1, 1, -1]])
    y = np.array([0.2, 0.5, 0.8, -0.1, 0.3])
    fits = [
        {
            "mean": [i / 10, 0.5],
            "prior": [[1.0, 0.1], [0.1, 0.25]],
            "sigma2": 0.2 + i / 10,
            "status": "SUPPORTED",
        }
        for i in range(3)
    ]
    means, covs = condition_source_fits(fits, x, y, kernel)
    for i, fit in enumerate(fits):
        direct = conditional_moments(
            fit["mean"], fit["prior"], x, fit["sigma2"] * kernel, y
        )
        np.testing.assert_allclose(means[i], direct["mean"], atol=1e-12)
        np.testing.assert_allclose(covs[i], direct["covariance"], atol=1e-12)
    empty = condition_source_fits(fits, np.empty((0, 2)), np.empty(0), np.empty((0, 0)))
    np.testing.assert_allclose(empty[0], [f["mean"] for f in fits])
    with pytest.raises(ValueError, match="unsupported"):
        condition_source_fits(fits + [fits[0] | {"status": "FAILED"}], x, y, kernel)
