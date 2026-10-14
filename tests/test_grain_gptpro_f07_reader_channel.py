"""Measured channel composition, ownership mixing and source isolation."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f07_channels import binary_law
from grain_research.gptpro_foundation_v3.f07_reader_channel import (
    fit_emission,
    reader_law,
)
from grain_research.gptpro_foundation_v3.f08_materials import material_bank


def source_rows():
    """Oracle-scored material fixture; actual learned scores are used by the runner."""
    return [
        {**r, "pole_score": r["pole"], "qualification_score": r["qualification"]}
        for r in material_bank()
        if r["role"] == "source"
    ]


def test_measured_oracle_reduces_to_missing_bernoulli():
    """Exactly half the declared mixed statements carry personal evidence."""
    emission = fit_emission(source_rows())
    np.testing.assert_allclose(emission, [[0.5, 0, 0.5], [0.5, 0.5, 0]])
    for z in (-1, 1):
        actual = reader_law([-0.2, 0.7], z, emission)
        expected = binary_law([-0.2, 0.7], z, 0.5, 0)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b)


def test_quoted_false_eligibility_adds_random_content():
    """Misattributed other-speaker content is not a perfectly personal label."""
    rows = source_rows()
    for r in rows:
        r["qualification_score"] = 1
    np.testing.assert_allclose(fit_emission(rows), [[0, 0.25, 0.75], [0, 0.75, 0.25]])


def test_asymmetric_channel_derivative():
    """General asymmetric emission derivatives match finite differences."""
    matrix = np.array([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
    theta = np.array([0.1, 0.5])
    h = 1e-5
    _, gradient = reader_law(theta, -1, matrix)
    numerical = np.column_stack(
        [
            (
                reader_law(theta + h * e, -1, matrix)[0]
                - reader_law(theta - h * e, -1, matrix)[0]
            )
            / (2 * h)
            for e in np.eye(2)
        ]
    )
    np.testing.assert_allclose(gradient, numerical, atol=1e-10)


def test_application_labels_cannot_fit_channel():
    """Held-out confusion cannot silently replace the source emission model."""
    rows = source_rows()
    rows[0]["role"] = "application"
    with pytest.raises(ValueError):
        fit_emission(rows)
