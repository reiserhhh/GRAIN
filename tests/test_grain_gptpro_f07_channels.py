"""Exact channel normalization, derivative and nuisance-information controls."""

import numpy as np

from grain_research.gptpro_foundation_v3.f07_channels import (
    binary_law,
    categorical_information,
    channel_information,
    efficient_slope_information,
    ordinal_law,
)


def test_categorical_derivatives_all_channels():
    theta = np.array([0.2, 0.6])
    for variant in ("base", "offtarget", "mnar", "ordinal"):
        law = (
            (lambda t: ordinal_law(t, -0.7))
            if variant == "ordinal"
            else (lambda t, variant=variant: binary_law(t, -0.7, 0.8, 0.2, variant))
        )
        p, derivative = law(theta)
        np.testing.assert_allclose(p.sum(), 1, atol=1e-14)
        np.testing.assert_allclose(derivative.sum(axis=0), 0, atol=1e-14)
        for step in (1e-4, 1e-5, 1e-6):
            numeric = np.column_stack(
                [
                    (
                        law(theta + np.eye(2)[j] * step)[0]
                        - law(theta - np.eye(2)[j] * step)[0]
                    )
                    / (2 * step)
                    for j in range(2)
                ]
            )
            np.testing.assert_allclose(derivative, numeric, atol=1e-9, rtol=1e-4)


def test_binary_formula_includes_changed_denominator():
    theta, z, c, epsilon = [0.0, 0.5], 1.0, 0.7, 0.25
    p = 1 / (1 + np.exp(-0.5))
    observed = epsilon + (1 - 2 * epsilon) * p
    expected = (
        c * (1 - 2 * epsilon) ** 2 * (p * (1 - p)) ** 2 / (observed * (1 - observed))
    )
    full = categorical_information(*binary_law(theta, z, c, epsilon))[1, 1]
    np.testing.assert_allclose(full, expected, atol=1e-14)
    shortcut = c * (1 - 2 * epsilon) ** 2 * p * (1 - p)
    assert abs(full - shortcut) > 1e-4


def test_missingness_can_inform_even_when_content_is_random():
    base = categorical_information(*binary_law([0.0, 0.5], 1.0, 0.9, 0.5))
    selected = categorical_information(*binary_law([0.0, 0.5], 1.0, 0.9, 0.5, "mnar"))
    np.testing.assert_array_equal(base, np.zeros((2, 2)))
    assert selected[1, 1] > 0


def test_fixed_condition_does_not_identify_slope_with_unknown_intercept():
    single = categorical_information(*binary_law([0.0, 0.5], 1.0, 0.9, 0.02))
    assert single[1, 1] > 0
    np.testing.assert_allclose(efficient_slope_information(single), 0, atol=1e-14)
    balanced = single + categorical_information(
        *binary_law([0.0, 0.5], -1.0, 0.9, 0.02)
    )
    assert efficient_slope_information(balanced) > 0


def test_flat_is_explicit_per_token_calibration():
    values = [
        channel_information([0.0, 0.5], level, "flat")["per_token"][1, 1]
        for level in range(5)
    ]
    np.testing.assert_allclose(values, np.repeat(values[0], 5), atol=1e-14)


def test_zero_presence_is_unobserved_not_informative_zero():
    p, d = binary_law([0.3, 0.5], 1.0, 0.0, 0.1)
    np.testing.assert_array_equal(p, [1, 0, 0])
    assert efficient_slope_information(categorical_information(p, d)) is None
