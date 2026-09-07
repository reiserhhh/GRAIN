#!/usr/bin/env python3
"""Exact finite statistical witnesses for query-accessible value variation."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.query_geometry import readout_geometry
from grain_research.vector_mechanisms.query_tilt import tilted_readout


def setup():
    rng = np.random.default_rng(732)
    return (
        rng,
        rng.normal(size=(30, 4)),
        rng.normal(size=(30, 3)),
        rng.normal(size=4) * 0.2,
        rng.normal(size=3),
    )


def test_slope_fisher_constraint_and_weighted_regression_identity():
    rng, k, v, q, beta = setup()
    result = readout_geometry(k, v, q, beta)
    d, eps = result["unit_fisher_direction"], 1e-5
    observed = (
        (tilted_readout(k, v, q + eps * d)[0] - tilted_readout(k, v, q - eps * d)[0])
        @ beta
        / (2 * eps)
    )
    assert observed == pytest.approx(np.sqrt(result["accessible_variance"]), rel=1e-7)
    assert d @ result["fisher"] @ d == pytest.approx(1)
    assert result["projection_identity_error"] < 1e-12
    assert 0 <= result["accessibility_fraction"] <= 1 + 1e-12
    for _ in range(30):
        candidate = rng.normal(size=4)
        candidate /= np.sqrt(candidate @ result["fisher"] @ candidate)
        assert result["gradient"] @ candidate <= observed + 1e-7


def test_nonsingular_key_coordinate_change_preserves_response_geometry():
    rng, k, v, q, beta = setup()
    transform = np.eye(4) + 0.1 * rng.normal(size=(4, 4))
    first = readout_geometry(k, v, q, beta)
    second = readout_geometry(k @ transform.T, v, np.linalg.solve(transform.T, q), beta)
    np.testing.assert_allclose(first["weights"], second["weights"], atol=1e-14)
    assert first["accessible_variance"] == pytest.approx(
        second["accessible_variance"], rel=1e-10
    )
    np.testing.assert_allclose(
        first["unit_fisher_direction"],
        transform.T @ second["unit_fisher_direction"],
        atol=1e-10,
    )


def test_degenerate_keys_and_perfect_linear_readout():
    _, k, _, q, _ = setup()
    result = readout_geometry(k, k, q, np.array([1.0, 0.0, 0.0, 0.0]))
    assert result["accessibility_fraction"] == pytest.approx(1)
    constant = readout_geometry(np.ones((30, 4)), k[:, :2], q, np.ones(2))
    assert constant["rank"] == 0
    assert constant["accessible_variance"] == 0
    np.testing.assert_array_equal(constant["unit_fisher_direction"], np.zeros(4))
