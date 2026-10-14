#!/usr/bin/env python3
"""Synthetic mechanism checks, with no model, labels or private text."""

import numpy as np
import pytest

from grain_research.construction import math_cycle_capacity as M


def test_cycle_projection_and_gradient_null():
    rng = np.random.default_rng(971)
    f = rng.normal(size=(8, 8, 3))
    odd, gradient, cycle = M.cycles(f)
    np.testing.assert_allclose(odd, gradient + cycle, atol=1e-13)
    np.testing.assert_allclose(cycle.sum(1), 0, atol=1e-13)
    np.testing.assert_allclose(np.sum(cycle * gradient, axis=(0, 1)), 0, atol=1e-13)
    p = rng.normal(size=(8, 3))
    np.testing.assert_allclose(M.cycles(p[:, None] - p[None, :])[2], 0, atol=1e-13)


def test_area_anchor_only_changes_gradient():
    rng = np.random.default_rng(972)
    x = rng.normal(size=(9, 4))
    a = M.area_bank(x, np.zeros(4))
    b = M.area_bank(x, rng.normal(size=4))
    np.testing.assert_allclose(M.cycles(a)[2], M.cycles(b)[2], atol=1e-13)
    np.testing.assert_allclose(a, -a.swapaxes(0, 1), atol=1e-13)


def test_full_affine_coordinates_span_every_cycle():
    n = 7
    x = np.eye(n)[:, :-1]
    a = M.area_bank(x, np.zeros(n - 1))
    basis, _, _ = M.orthogonal_basis(M.edge_columns(M.cycles(a)[2]))
    assert basis.shape[1] == (n - 1) * (n - 2) // 2
    rng = np.random.default_rng(973)
    result = M.decompose(a, rng.normal(size=(n, n, 4)))
    np.testing.assert_allclose(result["residual"], 0, atol=1e-12)


def test_low_rank_area_has_a_legal_invisible_witness():
    rng = np.random.default_rng(974)
    x = rng.normal(size=(8, 3))
    a = M.area_bank(x, np.zeros(3))
    f = rng.normal(size=(8, 8, 2))
    result = M.decompose(a, f)
    assert result["basis"].shape[1] == 3
    for q in range(2):
        plus, minus, scale = M.legal_witness(result["residual"][..., q])
        assert scale > 0 and min(plus.min(), minus.min()) > 0
        np.testing.assert_allclose(plus.sum(1), minus.sum(1), atol=1e-13)
        np.testing.assert_allclose(plus + plus.T, minus + minus.T, atol=1e-13)
        np.testing.assert_allclose(
            np.einsum("ij,ijq->q", plus - minus, a), 0, atol=1e-12
        )
        action = np.sum((plus - minus) * f[..., q])
        expected = 2 * scale * np.sum(result["residual"][..., q] ** 2)
        assert action > 0
        assert action == pytest.approx(expected, abs=1e-12)


def test_three_state_noncollinear_area_is_complete_but_collinear_is_not():
    f = np.arange(9.0).reshape(3, 3) ** 2
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert M.decompose(M.area_bank(x, np.zeros(2)), f[..., None])["basis"].shape[1] == 1
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert (
        M.decompose(M.area_bank(collinear, np.zeros(2)), f[..., None])["basis"].shape[1]
        == 0
    )


def test_invalid_support_and_noncycle_witness_fail():
    with pytest.raises(ValueError):
        M.cycles(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        M.legal_witness(np.array([[0.0, 1.0], [-1.0, 0.0]]))
