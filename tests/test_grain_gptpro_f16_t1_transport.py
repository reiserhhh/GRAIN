#!/usr/bin/env python3
"""Independent Gaussian/product-moment checks for pilot transport uncertainty."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3 import f16_t1_transport as m


def test_product_variance_equals_all_component_second_moments():
    rng = np.random.default_rng(3)
    kernels = {
        "weights": rng.normal(size=(5, 4)),
        "offsets": rng.normal(size=5),
        "conditional": rng.uniform(0.1, 1, size=5),
    }
    means = rng.normal(size=(7, 9, 4))
    result = m.integrate(kernels, means)
    for author in range(9):
        pairs = np.array(
            [
                kernels["offsets"][s] + kernels["weights"][s] @ means[j, author]
                for s in range(5)
                for j in range(7)
            ]
        )
        variance = kernels["conditional"].mean() + np.mean(pairs**2) - pairs.mean() ** 2
        np.testing.assert_allclose(result["score"][author], pairs.mean(), atol=1e-12)
        np.testing.assert_allclose(result["variance"][author], variance, atol=1e-12)


def test_identical_calibrations_add_zero_pilot_uncertainty():
    kernels = {"weights": np.eye(4), "offsets": np.zeros(4), "conditional": np.ones(4)}
    means = np.repeat(np.arange(8).reshape(1, 2, 4), 20, axis=0)
    result = m.integrate(kernels, means)
    np.testing.assert_array_equal(result["pilot"], 0)
    assert np.all(result["source"] > 0)


def test_cell_means_and_invalid_shapes():
    z = np.tile(m.CONDITIONS, 2)
    y = np.arange(16).reshape(2, 8)
    np.testing.assert_array_equal(m.cell_means(y, z), [[2, 3, 4, 5], [10, 11, 12, 13]])
    with pytest.raises(ValueError):
        m.cell_means(y[:, :2], z[:2])
    with pytest.raises(ValueError):
        m.integrate({}, np.ones((3, 4)))


def test_bootstraps_reproducible_whole_author_indices():
    a = m.bootstrap_indices(20260915)
    np.testing.assert_array_equal(a, m.bootstrap_indices(20260915))
    assert a.shape == (200, 60) and a.min() == 0 and a.max() == 59
