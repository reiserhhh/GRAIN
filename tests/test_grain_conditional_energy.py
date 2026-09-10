#!/usr/bin/env python3
"""Finite witnesses for cancellation, conditional moments and score links."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.action_geometry import output_gradient
from grain_research.vector_mechanisms.conditional_energy import (
    energy_gap,
    fit_distribution,
    mixture_moments,
    mixture_weights,
    score_hessian,
    score_mixture,
)


def test_opposite_actions_have_zero_mean_nonzero_energy():
    m = mixture_moments([[0, 0.5, 0.5]], [[1, -1], [-1, 1]])
    np.testing.assert_allclose(m["mean"], 0)
    np.testing.assert_allclose(m["energy"], 2)
    np.testing.assert_allclose(m["variance_trace"], 2)


def test_exact_energy_identity_and_psd_with_zero_atom():
    rng = np.random.default_rng(2501)
    y = rng.normal(size=(12, 5))
    w = rng.uniform(size=(8, 13))
    w /= w.sum(1, keepdims=True)
    m = mixture_moments(w, y)
    np.testing.assert_allclose(
        m["energy"], m["mean_energy"] + m["variance_trace"], atol=1e-14
    )
    assert np.linalg.eigvalsh(m["covariance"]).min() > -1e-14


def test_arbitrary_predictor_cross_term_is_not_zero():
    y, p = np.array([[1.0, 2.0]]), np.array([[2.0, 4.0]])
    m = energy_gap(y, p)
    assert m["energy_gap"][0] == -15
    assert m["residual_energy"][0] == 5
    assert m["cross_term"][0] == -20
    assert m["identity_error"][0] == 0


def test_swap_zero_and_training_arrays_immutable():
    rng = np.random.default_rng(44)
    a, b, y = [rng.normal(size=(6, d)) for d in (4, 4, 3)]
    model = fit_distribution(a, b, y, scale=2)
    test_a, test_b = rng.normal(size=(2, 4)), rng.normal(size=(2, 4))
    first = mixture_weights(test_a, test_b, model)
    np.testing.assert_allclose(first, mixture_weights(test_b, test_a, model))
    zero = mixture_weights(test_a * 0, test_b, model)
    np.testing.assert_array_equal(zero[:, 0], 1)
    np.testing.assert_array_equal(zero[:, 1:], 0)
    np.testing.assert_array_equal(mixture_moments(zero, y)["energy"], 0)
    model_before = model["targets"].copy()
    y[:] = 9000
    np.testing.assert_array_equal(model["targets"], model_before)
    np.testing.assert_array_equal(mixture_weights(test_a, test_b, model), first)


def test_score_hessian_agrees_with_gradient_differences():
    z, values = np.array([0.2, -1.4, 0.9]), np.array([0.0, 0.5, 1.0])
    eps = 1e-5
    columns = [
        (output_gradient(z + eps * d, values) - output_gradient(z - eps * d, values))
        / (2 * eps)
        for d in np.eye(3)
    ]
    h = score_hessian(z, values)
    np.testing.assert_allclose(h, np.array(columns).T, atol=1e-10)
    np.testing.assert_allclose(h, h.T, atol=1e-15)
    np.testing.assert_allclose(h @ np.ones(3), 0, atol=1e-15)


def test_covariance_correction_approximates_small_mixture_effect():
    y = np.array([[1, -1], [-1, 1]], float) * 0.002
    result = score_mixture([0.3, -0.5], [0, 1], [0.2, 0.4, 0.4], y)
    assert abs(result["mixture_minus_mean_score"]) > 1e-8
    assert abs(result["higher_order_remainder"]) < 1e-10


def test_permutation_invariance_of_reference_atoms():
    y = np.array([[2, 0], [1, -3], [-2, 1]], float)
    w = np.array([[0.1, 0.2, 0.3, 0.4]])
    a = mixture_moments(w, y)
    b = mixture_moments(w[:, [0, 3, 1, 2]], y[[2, 0, 1]])
    for key in ("mean", "second", "covariance", "energy", "mean_energy"):
        np.testing.assert_allclose(a[key], b[key])


@pytest.mark.parametrize("weights", [[[0, -1, 2]], [[0, 1, 1]], [[np.nan, 0.5, 0.5]]])
def test_invalid_mixtures_rejected(weights):
    with pytest.raises(ValueError):
        mixture_moments(weights, [[1], [-1]])


def test_missing_support_fails_explicitly():
    with pytest.raises(ValueError):
        fit_distribution(np.zeros((3, 2)), np.ones((3, 2)), np.ones((3, 2)), scale=1)


def test_first_moment_does_not_identify_second_moment():
    a = mixture_moments([[0, 0.5, 0.5]], [[1], [-1]])
    b = mixture_moments([[0, 0.5, 0.5]], [[3], [-3]])
    np.testing.assert_allclose(a["mean"], b["mean"])
    assert b["energy"][0] == 9 * a["energy"][0]


def test_condition_refinement_separates_mean_shift_from_remaining_variance():
    coarse = mixture_moments([[0, 0.5, 0.5]], [[2], [-2]])
    fine_means = np.array([2.0, -2.0])
    fine_variances = np.zeros(2)
    between = np.mean((fine_means - coarse["mean"][0, 0]) ** 2)
    assert coarse["variance_trace"][0] == fine_variances.mean() + between


def test_more_conditioning_reduces_average_not_every_pointwise_variance():
    # Ninety percent is a fixed zero; a rare stratum has two opposing values.
    coarse = mixture_moments([[0.9, 0.05, 0.05]], [[10], [-10]])
    rare = mixture_moments([[0, 0.5, 0.5]], [[10], [-10]])
    assert rare["variance_trace"][0] > coarse["variance_trace"][0]
    assert coarse["variance_trace"][0] == 0.1 * rare["variance_trace"][0]
