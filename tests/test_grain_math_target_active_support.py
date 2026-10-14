#!/usr/bin/env python3
"""Synthetic checks for the frozen HM36 source support protocol."""

import numpy as np
import pytest

from grain_research.construction.math_target_active_support import (
    action_parts,
    metric_projection,
    random_basis,
    signed_metrics,
    source_basis,
)


def test_rank_deficiency_and_column_units():
    x = np.array([[1.0, 3.0, 0], [0, 0, 2], [0, 0, 0]])
    a = source_basis(x, ["source"] * 3)
    b = source_basis(x * [7, 0.1, 12], ["source"] * 3)
    assert a["rank"] == b["rank"] == 2
    np.testing.assert_allclose(a["basis"] @ a["basis"].T, b["basis"] @ b["basis"].T)


def test_exact_zero_columns_explicitly_recorded():
    a = source_basis(np.zeros((4, 2)), ["source"] * 2)
    assert a["rank"] == 0 and not a["nonzero"].any()
    assert a["basis"].shape == (4, 0)


def test_source_isolation():
    with pytest.raises(ValueError, match="application"):
        source_basis(np.eye(3), ["source", "application", "source"])


def test_random_dimension_rank_seed():
    a = random_basis(12, 3, 1701)
    np.testing.assert_array_equal(a, random_basis(12, 3, 1701))
    np.testing.assert_allclose(a.T @ a, np.eye(3), atol=1e-14)
    assert not np.array_equal(a, random_basis(12, 3, 1702))
    with pytest.raises(ValueError):
        random_basis(2, 3, 1701)


def test_signed_cancellation_not_clipped():
    u = np.array([[1.0], [0.0]])
    parts = action_parts(u, np.array([[-2.0, 3.0]]), np.ones(2))
    m = signed_metrics(
        np.array([list(parts[k]) for k in ("full", "supported", "missing")]).T
    )
    assert m["missing_over_full"] == 3 and m["supported_over_full"] == 2
    assert m["cross_over_full_energy"] == -12 and m["identity_error"] == 0


def test_covector_displacement_orthogonal_energy():
    u = random_basis(7, 2, 17)
    v, d = np.arange(14).reshape(2, 7), np.arange(7)
    p = action_parts(u, v, d)
    np.testing.assert_allclose(p["full"], p["supported"] + p["missing"])
    assert p["displacement_supported_energy"] <= p["displacement_energy"] + 1e-12
    assert (p["covector_supported_energy"] <= p["covector_energy"] + 1e-12).all()


def test_nonorthogonal_chart_requires_transported_metric():
    x = np.array([[1.0], [2.0], [0.0]])
    d = np.array([3.0, 1.0, 2.0])
    transform = np.array([[2.0, 1.0, 0], [0, 1.0, 1.0], [0, 0, 0.5]])
    inverse = np.linalg.inv(transform)
    g = inverse.T @ inverse
    physical = metric_projection(x, d, np.eye(3))
    chart = metric_projection(transform @ x, transform @ d, g)
    np.testing.assert_allclose(chart, transform @ physical, atol=1e-13)
    covector = np.array([0.5, -0.7, 0.1])
    np.testing.assert_allclose(covector @ physical, (inverse.T @ covector) @ chart)


def test_target_support_can_ignore_large_inactive_variation():
    u = source_basis(np.array([[1.0], [0]]), ["source"])["basis"]
    p = action_parts(u, np.array([[2.0, 0]]), np.array([1.0, 1000.0]))
    assert p["full"][0] == p["supported"][0] == 2
    assert p["displacement_supported_energy"] / p["displacement_energy"] < 1e-5
