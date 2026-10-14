"""Synthetic tests for the GP06 cold-construction primitives."""

import numpy as np

from grain_research.measurement_discovery.construction import (
    affine_standardize,
    make_bundle,
    next_observation_probability,
    theta_response,
    transport_standardization,
)


def test_theta_and_next_probability():
    assert np.isclose(theta_response(.8, .2), .3)
    assert next_observation_probability(1, 0, [.1, .2, .7, .9]) == .7


def test_affine_transport_preserves_standard_score():
    mean, scale = transport_standardization(2, 3, 2, 3)
    assert np.isclose(affine_standardize(2, 2, 3), affine_standardize(7, mean, scale))


def test_bundle_has_explicit_support():
    bundle = make_bundle("t", "x", "unit", {"mean": 0}, "reader", "observed only")
    assert bundle["support"] == "observed only"
