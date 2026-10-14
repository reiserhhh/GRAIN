#!/usr/bin/env python3
"""Known derivative, orthogonality and routing alternatives; no private data."""

import numpy as np
import pytest

from grain_research.construction import math_binding_router as M


def test_mixed_hessian_and_transpose_are_not_output_hessian():
    import torch

    h = np.array([[1.0, 2.0], [-3.0, 4.0]])
    a = torch.tensor(h, dtype=torch.float32)
    function = lambda x, y: x @ a @ y + (x @ x) * 3 + (y @ y) * 7
    _, observed = M.mixed_product(function, 2, np.eye(2))
    _, transpose = M.mixed_product(function, 2, np.eye(2), transpose=True)
    np.testing.assert_allclose(observed, h)
    np.testing.assert_allclose(transpose, h.T)


def test_randomized_factorization_known_low_rank():
    rng = np.random.default_rng(9)
    left, right = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    h = left @ right.T
    omega = M.gaussian_basis(20, 7, 13)
    y = h @ omega
    q = np.linalg.qr(y)[0]
    u, singular, v = M.compressed_svd(y, h.T @ q, 3)
    np.testing.assert_allclose((u * singular[:3]) @ v.T, h, atol=1e-12)


def test_projection_swap_is_involution_and_preserves_complement():
    basis = M.gaussian_basis(12, 4, 18)
    left, right = np.arange(12.0), np.arange(12.0)[::-1]
    delta = M.swap_delta(left, right, basis)
    second = M.swap_delta(left + delta, right - delta, basis)
    np.testing.assert_allclose(second, -delta, atol=1e-12)
    np.testing.assert_allclose(basis.T @ (left + delta), basis.T @ right)
    np.testing.assert_allclose((np.eye(12) - basis @ basis.T) @ delta, 0, atol=1e-12)


def test_equal_norm_control_and_invalid_basis():
    basis = M.gaussian_basis(12, 4, 3)
    delta = M.swap_delta(np.zeros(12), np.ones(12), basis, match_norm=2.3)
    assert np.linalg.norm(delta) == pytest.approx(2.3)
    with pytest.raises(ValueError):
        M.swap_delta(np.zeros(12), np.ones(12), basis * 2)


def test_natural_swap_changes_person_not_common_or_third_person():
    baseline = np.array([[0.8, 0.7], [0.2, 0.3], [0.5, 0.5]])
    natural = baseline[[1, 0, 2]]
    account = M.routing_accounts(baseline, natural, natural)
    assert account["towards_natural_fraction"] == pytest.approx(1)
    assert account["common_norm"] == 0
    assert account["untouched_norm"] == 0


def test_large_common_shift_is_not_assignment_routing():
    baseline = np.array([[0.8], [0.2], [0.5]])
    account = M.routing_accounts(baseline, baseline + 0.1, baseline[[1, 0, 2]])
    assert account["towards_natural_fraction"] == pytest.approx(0)
    assert account["untouched_norm"] == pytest.approx(0.1)


def test_shape_guard():
    with pytest.raises(ValueError):
        M.routing_accounts(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 4)))
