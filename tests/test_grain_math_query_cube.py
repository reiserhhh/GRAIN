#!/usr/bin/env python3
"""HM32 exact cube support, character aliasing and postpatch differentiation."""

import numpy as np
import pytest
import torch

from grain_research.construction import math_position_support as P
from grain_research.construction import math_query_cube as U
from scripts.verify_grain_math_query_cube import product_parts


def test_characters_orthogonal_and_group_law():
    chars = U.characters()
    np.testing.assert_array_equal(chars @ chars.T, 8 * np.eye(8))
    for a in range(8):
        for b in range(8):
            np.testing.assert_array_equal(chars[a] * chars[b], chars[a ^ b])


def test_natural_diagonal_cannot_separate_r_and_q():
    diagonal = [i for i, (_, r, q) in enumerate(U.BITS) if r == q]
    chars = U.characters()
    for mask in range(8):
        np.testing.assert_array_equal(chars[mask, diagonal], chars[mask ^ 3, diagonal])
    assert np.linalg.matrix_rank(chars[:, diagonal]) == 4
    assert np.linalg.matrix_rank(chars) == 8


@pytest.mark.parametrize("mask", range(8))
def test_single_character_energy_and_bounds(mask):
    x = U.characters()[mask].reshape(2, 2, 2)
    coefficients, stats = U.split(x)
    np.testing.assert_array_equal(coefficients, np.eye(8)[mask])
    assert stats[U.LABELS[mask] + "_energy"] == 1
    assert stats["HQ_only_mse_bound"] == int(bool(mask & 2))
    assert stats["HR_only_mse_bound"] == int(bool(mask & 1))


def test_vector_reconstruction_parseval_and_translation():
    x = np.random.default_rng(32).normal(size=(2, 2, 2, 4, 3))
    coefficients, stats = U.split(x)
    assert stats["reconstruction_error"] < 1e-14
    assert stats["energy_identity_error"] < 1e-13
    shifted, shifted_stats = U.split(x[::-1])
    np.testing.assert_allclose(shifted[:4], coefficients[:4])
    np.testing.assert_allclose(shifted[4:], -coefficients[4:])
    assert stats["HQ_only_mse_bound"] == pytest.approx(
        shifted_stats["HQ_only_mse_bound"]
    )


def test_bound_is_conditional_variance_not_fitted_prediction():
    x = np.random.default_rng(7).normal(size=(2, 2, 2, 3))
    _, stats = U.split(x)
    hq = x.mean(axis=1, keepdims=True)
    hr = x.mean(axis=2, keepdims=True)
    h = x.mean(axis=(1, 2), keepdims=True)
    assert stats["H_only_mse_bound"] == pytest.approx(np.sum((x - h) ** 2) / 8)
    assert stats["HQ_only_mse_bound"] == pytest.approx(np.sum((x - hq) ** 2) / 8)
    assert stats["HR_only_mse_bound"] == pytest.approx(np.sum((x - hr) ** 2) / 8)


def test_q_only_versus_r_only_action_counterexamples():
    q = U.characters()[1].reshape(2, 2, 2)
    r = U.characters()[2].reshape(2, 2, 2)
    assert all(c["difference_norm"] == 0 for c in U.contrasts(q) if c["vary"] == "R")
    assert all(c["difference_norm"] > 0 for c in U.contrasts(r) if c["vary"] == "R")
    assert U.split(q)[1]["HQ_only_mse_bound"] == 0
    assert U.split(r)[1]["HQ_only_mse_bound"] == 1


def test_patched_leaf_h_gradient_depends_on_q_with_r_fixed():
    original = torch.arange(24.0, requires_grad=True).reshape(1, 8, 3)
    receiver_h = original[0, :4].detach().numpy()
    receiver_q = original[0, -1].detach().numpy()
    donor_h, donor_q = receiver_h + 2, receiver_q + 3
    leaf, audit = U.replace_leaf(
        original, [0, 1, 2, 3], receiver_h, receiver_q, donor_h, donor_q, True
    )
    g = torch.autograd.grad((leaf[0, :4] * leaf[0, -1]).sum(), leaf)[0][0, :4]
    np.testing.assert_array_equal(g.numpy(), np.broadcast_to(donor_q, (4, 3)))
    np.testing.assert_array_equal(
        leaf.detach().numpy()[0, 4:7], original.detach().numpy()[0, 4:7]
    )
    assert leaf.is_leaf and all(v == 0 for v in audit.values())


def test_changed_receiver_query_rejected():
    with pytest.raises(ValueError):
        U.replace_leaf(
            torch.zeros(1, 8, 3),
            [0, 1, 2, 3],
            np.zeros((4, 3)),
            np.ones(3),
            np.zeros((4, 3)),
            np.zeros(3),
            False,
        )


def test_zero_energy_and_invalid_cube():
    assert U.split(np.zeros((2, 2, 2)))[1]["HQ_only_relative_rmse_bound"] is None
    with pytest.raises(ValueError):
        U.split(np.zeros((2, 2, 1)))
    with pytest.raises(ValueError):
        U.split(np.full((2, 2, 2), np.nan))


def test_alternate_product_projection_agrees_with_fourier():
    x = np.random.default_rng(321).normal(size=(2, 2, 2, 4, 3))
    primary, stats = U.split(x)
    alternate, alternate_stats = product_parts(x)
    np.testing.assert_allclose(primary, alternate, atol=1e-14)
    for name, value in stats.items():
        assert value == pytest.approx(alternate_stats[name], abs=1e-13)


def test_same_donor_physical_delta_is_identical_across_r_and_q():
    h = np.random.default_rng(3).normal(size=(4, 3)).astype(np.float32)
    direction = np.ones_like(h)
    expected = P.physical_patches(h, direction)["half_delta"]
    for r, q in ((0, 0), (0, 1), (1, 0), (1, 1)):
        natural = torch.ones(1, 8, 3) * (r + 1)
        leaf, _ = U.replace_leaf(
            natural,
            [0, 1, 2, 3],
            natural[0, :4].numpy(),
            natural[0, -1].numpy(),
            h,
            np.full(3, q, np.float32),
            False,
        )
        actual = P.physical_patches(leaf[0, :4].numpy(), direction)["half_delta"]
        np.testing.assert_array_equal(actual, expected)


def test_design_expands_exact_query_tokens_without_crossing_roles():
    natural = [
        {"unit": i, "input_ids": [i, 10, 11], "question_start": 1} for i in range(2)
    ]
    rows = []
    for i in range(2):
        for j in range(2):
            rows.append(
                {
                    "unit": len(rows),
                    "pair": 0,
                    "wording": 0,
                    "query": "q",
                    "donor_side": i,
                    "receiver_side": j,
                    "receiver_unit": j,
                    "input_ids": natural[j]["input_ids"],
                    "question_start": 1,
                    "role": "source",
                }
            )
    cube = U.design(rows, natural)
    assert len(cube) == 8 and sum(r["inherited"] for r in cube) == 4
    assert {
        (r["donor_side"], r["receiver_side"], r["query_side"]) for r in cube
    } == set(U.BITS)
    assert all(
        r["role"] == "source" and r["query_endpoint_token_id"] == 11 for r in cube
    )
    natural[1] = {**natural[1], "input_ids": [1, 10, 12]}
    with pytest.raises(ValueError, match="query suffix"):
        U.design(rows, natural)
