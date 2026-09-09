#!/usr/bin/env python3
"""Conditional Fourier differences, parity, higher moments, and training boundary."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.kernel_action import (
    ARMS,
    contrast_features,
    fit_action,
    predict_action,
    training_geometry,
)


def inputs():
    rng = np.random.default_rng(3)
    return (
        rng.normal(size=(7, 4)),
        rng.normal(size=(7, 4)),
        np.linspace(-1, 1, 7),
        rng.normal(size=4),
        np.zeros(4),
        np.ones(4),
    )


def test_fourier_parts_reproduce_direct_feature_difference():
    c, d, pos, eta, m, s = inputs()
    for arm in ["fourier_bag", "fourier_position"]:
        odd, even = contrast_features(c, d, pos, eta, m, s, arm)
        theta = c + (pos[:, None] * eta if arm.endswith("position") else 0)
        phi = lambda t: np.concatenate([np.cos(t).mean(0), np.sin(t).mean(0)]) / 2
        np.testing.assert_allclose(odd + even, phi(theta + d) - phi(theta), atol=1e-14)
        np.testing.assert_allclose(-odd + even, phi(theta - d) - phi(theta), atol=1e-14)


def test_every_arm_has_zero_and_exact_parity():
    c, d, pos, eta, m, s = inputs()
    for arm in ARMS:
        odd, even = contrast_features(c, d, pos, eta, m, s, arm)
        other = contrast_features(c, -d, pos, eta, m, s, arm)
        np.testing.assert_allclose(other[0], -odd, atol=1e-14)
        np.testing.assert_allclose(other[1], even, atol=1e-14)
        zero = contrast_features(c, np.zeros_like(d), pos, eta, m, s, arm)
        np.testing.assert_array_equal(zero, np.zeros((2, 8)))


def test_fourier_distinguishes_distributions_with_identical_first_three_moments():
    a = np.array([-1, -1, 1, 1.0])[:, None]
    b = np.array([-np.sqrt(2), 0, 0, np.sqrt(2)])[:, None]
    for degree in [1, 2, 3]:
        assert np.mean(a**degree) == pytest.approx(np.mean(b**degree))
    feature = lambda d, arm: contrast_features(
        np.zeros_like(d), d, np.zeros(4), np.zeros(1), np.zeros(1), np.ones(1), arm
    )
    np.testing.assert_allclose(
        feature(a, "taylor3_bag"), feature(b, "taylor3_bag"), atol=1e-14
    )
    assert (
        np.linalg.norm(
            np.asarray(feature(a, "fourier_bag")) - feature(b, "fourier_bag")
        )
        > 0.03
    )


def test_paired_loss_is_exactly_odd_plus_even_loss():
    rng = np.random.default_rng(17)
    o, e, po, pe = rng.normal(size=(4, 20))
    paired = ((o + e - po - pe) ** 2 + (-o + e + po - pe) ** 2) / 2
    np.testing.assert_allclose(paired, (o - po) ** 2 + (e - pe) ** 2, atol=1e-14)


def test_held_moments_do_not_change_geometry_or_fixed_action():
    rng = np.random.default_rng(5)
    means = rng.normal(size=(12, 4))
    var = np.abs(rng.normal(size=(12, 4)))
    train = np.arange(12) < 9
    m, s = training_geometry(means, var, train)
    means[~train] = np.nan
    var[~train] = np.inf
    m2, s2 = training_geometry(means, var, train)
    np.testing.assert_array_equal(m, m2)
    np.testing.assert_array_equal(s, s2)
    x = rng.normal(size=(12, 4))
    y = rng.normal(size=12)
    model = fit_action(x[train], y[train])
    before = predict_action(x[~train], model)
    y[~train] += 10000
    repeated = fit_action(x[train], y[train])
    np.testing.assert_array_equal(before, predict_action(x[~train], repeated))
    np.testing.assert_array_equal(predict_action(np.zeros((1, 4)), model), [0.0])


def test_bad_geometry_and_features_fail():
    c, d, pos, eta, m, _s = inputs()
    with pytest.raises(ValueError):
        contrast_features(c, d, pos, eta, m, np.zeros(4), "fourier_bag")
    with pytest.raises(ValueError):
        training_geometry(c, d, np.zeros(len(c), bool))
    with pytest.raises(ValueError):
        fit_action(c, np.ones(2))
