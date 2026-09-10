#!/usr/bin/env python3
"""Input-null witnesses, orthogonal geometry and signed native responses."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.pooled_feature_null_response import (
    PARTS,
    decompose,
    response_account,
    responses,
)


def test_uniform_gradient_has_no_allocation_response():
    rng = np.random.default_rng(59)
    d = rng.normal(size=(12, 5))
    g = np.broadcast_to(np.arange(5.0), d.shape)
    v, _ = responses(g, d, np.eye(5)[:, :2])
    assert abs(v["allocation"]) < 1e-12
    assert sum(v[n] for n in PARTS) == pytest.approx(v["native"])


def test_token_allocation_can_be_invisible_but_active():
    d = np.array([[1.0, 0.0], [-1.0, 0.0]])
    v, geo = responses(d, d, np.eye(2))
    assert v["visible"] == 0
    assert v["allocation"] == 2
    assert geo["feature_derivative_norm"] == 0
    assert geo["allocation_feature_error"] == 0


def test_basis_outside_mean_can_be_invisible_but_active():
    d = np.array([[0.0, 2.0], [0.0, 2.0]])
    v, geo = responses(d, d, np.eye(2)[:, :1])
    assert v["visible"] == 0
    assert v["allocation"] == 0
    assert v["outside"] == 8
    assert geo["outside_feature_error"] == 0


def test_scalar_response_components_can_cancel_despite_orthogonal_vectors():
    d = np.array([[2.0, 0.0], [0.0, 0.0]])
    g = np.array([[0.0, 0.0], [2.0, 0.0]])
    v, geo = responses(g, d, np.eye(2))
    assert v["visible"] == 2 and v["allocation"] == -2
    assert v["native"] == 0
    assert geo["visible_allocation_inner"] == 0
    s = response_account([[v[k] for k in ("native", *PARTS)]])
    assert np.isnan(s["visible_projection"])
    assert s["cross_visible_allocation"] == -8
    assert s["response_energy_identity_error"] == 0


def test_orthogonal_coordinate_change_preserves_components():
    rng = np.random.default_rng(9)
    d, g = rng.normal(size=(2, 16, 6))
    u = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    rotation = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    a, ga = responses(g, d, u)
    b, gb = responses(g @ rotation, d @ rotation, rotation.T @ u)
    for n in ("native", *PARTS):
        assert a[n] == pytest.approx(b[n], abs=1e-11)
    assert ga["total_energy"] == pytest.approx(gb["total_energy"])


def test_zero_direction_and_energy_identity():
    p, g = decompose(np.zeros((5, 3)), np.eye(3)[:, :2])
    assert sum(np.sum(v * v) for v in p) == 0
    assert g["energy_identity_error"] == 0
    s = response_account(np.zeros((4, 4)))
    assert s["native_rms"] == 0 and np.isnan(s["null_squared_ratio"])


def test_invalid_basis_and_gradient_rejected():
    with pytest.raises(ValueError, match="orthonormal"):
        decompose(np.ones((4, 3)), 2 * np.eye(3))
    with pytest.raises(ValueError, match="gradient"):
        responses(np.zeros((3, 3)), np.zeros((4, 3)), np.eye(3))
