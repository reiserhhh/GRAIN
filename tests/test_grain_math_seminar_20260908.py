#!/usr/bin/env python3
"""Check the finite witnesses before using them to guide the next design."""

import numpy as np

from scripts.check_grain_math_seminar_20260908 import (
    conditional_expectation_witness,
    coordinate_witness,
    pairing_witness,
    path_witness,
    variance_witness,
)


def test_variance_is_not_functional_importance():
    w = variance_witness()
    assert w["pca_retained_variance_fraction"] > 0.999
    assert w["pca_response_mse"] == 1
    assert w["functional_response_mse"] == 0


def test_pairing_carries_information_beyond_two_marginals():
    w = pairing_witness()
    assert w["gradient_second_moment_difference"] == 0
    assert w["paired_same_response"] == [1.0, 1.0]
    assert w["paired_crossed_response"] == [0.0, 0.0]


def test_dual_coordinate_transform_preserves_function():
    w = coordinate_witness()
    assert w["paired_effect_max_error"] < 1e-12
    assert w["product_spectrum_max_error"] < 1e-10
    assert w["euclidean_functional_cosine_original"] == 0
    assert w["euclidean_functional_cosine_rescaled"] < -0.99


def test_area_is_not_scalar_gradient_holonomy():
    w = path_witness()
    assert w["same_endpoint"]
    assert np.allclose(w["signed_areas"], [0.5, -0.5])
    assert np.allclose(w["potential_integrals"], [1.0, 1.0])
    assert w["closed_gradient_integral"] == 0


def test_nested_information_explains_interaction_residual():
    w = conditional_expectation_witness()
    assert w["population_squared_errors"] == [5.25, 4.25, 0.25, 0.0]
    assert w["explained_increment_energies"] == [1.0, 4.0, 0.25]
    assert w["pythagorean_identity_max_error"] == 0.0
