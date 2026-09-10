#!/usr/bin/env python3
"""Independent source spectrum examples, including weak features with real signal."""

import numpy as np
import pytest

from scripts.report_grain_paired_response_objective import spectral_account


def test_resolved_capacity_and_ridge_have_different_limits():
    a = np.diag([1.0, 0.01, 0.0])
    target = np.array([0.6, 0.8, 0.0])
    g, r = a.T @ a, a.T @ target
    out = spectral_account(g, r, np.zeros(3), 0.01)
    assert out["resolved_rank"] == 2
    assert out["resolved_target_energy"] == pytest.approx(1)
    assert out["resolved_minimum_loss"] == pytest.approx(0, abs=1e-12)
    assert out["resolved_minimum_ridge_cost"] > 60
    assert out["ridge_only_delta_loss"] > 0.6
    expected = np.linalg.solve(g + 0.01 * np.eye(3), r)
    assert out["ridge_only_delta_loss"] == pytest.approx(
        np.sum((a @ expected - target) ** 2)
    )
    assert out["spectral_objective_error"] < 1e-12


def test_external_target_and_coordinate_rotation():
    rng = np.random.default_rng(66)
    a = np.diag([1.0, 0.5, 0.0])
    t = np.array([0.5, 0.5, np.sqrt(0.5)])
    g, r = a.T @ a, a.T @ t
    c = rng.normal(size=3)
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    first = spectral_account(g, r, c, 0.1)
    rotated = spectral_account(q.T @ g @ q, q.T @ r, q.T @ c, 0.1)
    assert first["resolved_target_energy"] == pytest.approx(0.5)
    for key in (
        "resolved_target_energy",
        "current_delta_loss",
        "ridge_only_delta_loss",
    ):
        assert first[key] == pytest.approx(rotated[key], abs=1e-10)


def test_local_token_scalar_and_normalized_pooling_have_distinct_cross_response():
    """Holding last fixed, local token sums cannot change an untouched token's gradient."""
    from test_grain_full_context_geometry import toy

    from grain_research.vector_mechanisms import full_context_geometry as f

    rng, bodies, lasts, _, _, model = toy()
    body = bodies[0]
    direction = np.zeros_like(body)
    direction[: len(body) // 2] = rng.normal(size=direction[: len(body) // 2].shape)
    left = f.predict(body + 0.03 * direction, lasts[0], model)[1][len(body) // 2 :]
    right = f.predict(body - 0.03 * direction, lasts[0], model)[1][len(body) // 2 :]
    np.testing.assert_array_equal(left, right)

    def attention_gradient(x):
        weight = np.exp(x - np.max(x))
        weight /= weight.sum()
        value = weight @ x
        return weight * (1 + x - value)

    x = np.array([0.2, 0.8])
    changed = attention_gradient(x + np.array([0.03, 0.0]))[1]
    original = attention_gradient(x)[1]
    assert abs(changed - original) > 1e-4
