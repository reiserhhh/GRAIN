#!/usr/bin/env python3
"""Synthetic witnesses for distribution versus mean regression."""

import numpy as np

from scripts.run_grain_distribution_student import (
    bandwidth,
    block_normalize,
    fixed_projection,
    fourier_features,
    raw_features,
)


def test_same_mean_different_distribution_survives_kernel_mean():
    bags = np.array([[[-1.0], [1.0]], [[-2.0], [2.0]]])
    projected, phase = fixed_projection(bags, 1024)
    mean = fourier_features(projected, phase, 1, "mean")
    distribution = fourier_features(projected, phase, 1, "bag")
    assert np.allclose(mean[0], mean[1])
    assert np.linalg.norm(distribution[0] - distribution[1]) > 0.3


def test_random_features_approximate_gaussian_kernel():
    bags = np.array([[[0.0]], [[1.0]]])
    projected, phase = fixed_projection(bags, 20000)
    features = fourier_features(projected, phase, 1, "mean")
    assert abs(features[0] @ features[1] - np.exp(-0.5)) < 0.025


def test_map_is_comment_order_invariant_but_not_mean_only():
    rng = np.random.default_rng(3)
    bags = rng.normal(size=(6, 8, 12))
    p, phase = fixed_projection(bags)
    assert np.allclose(
        fourier_features(p, phase, 1, "bag"),
        fourier_features(p[:, ::-1], phase, 1, "bag"),
        atol=1e-6,
    )


def test_test_authors_do_not_set_bandwidth_or_training_geometry():
    rng = np.random.default_rng(6)
    bags = rng.normal(size=(8, 4, 10))
    train = np.arange(5)
    p, phase = fixed_projection(bags)
    raw, sigma = raw_features(bags, p, phase, train)
    z, params = block_normalize(raw, train)
    altered = bags.copy()
    altered[6:] *= 100
    p2, phase2 = fixed_projection(altered)
    raw2, sigma2 = raw_features(altered, p2, phase2, train)
    z2, params2 = block_normalize(raw2, train)
    assert sigma == sigma2
    for name in params:
        assert np.allclose(params[name], params2[name])
    for name in z:
        assert np.allclose(z[name][:6], z2[name][:6])
        assert abs(np.mean(np.sum(z[name][train] ** 2, axis=1)) - 1) < 1e-6


def test_degenerate_bandwidth_has_finite_fallback():
    assert bandwidth(np.ones((8, 4))) == 1
