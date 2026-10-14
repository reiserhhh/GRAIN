#!/usr/bin/env python3
"""HM33 group support, mapped states, known carriers and finite oracle bounds."""

import itertools

import numpy as np
import pytest
import torch

from grain_research.construction import math_position_support as D
from grain_research.construction import math_query_context as U
from grain_research.construction import math_query_cube as C
from scripts.verify_grain_math_query_context import product_parts


def test_full_group_characters_and_old_diagonal_alias():
    chars = U.characters()
    np.testing.assert_array_equal(chars @ chars.T, 16 * np.eye(16))
    diagonal = [i for i, (_, b, p, _) in enumerate(U.BITS) if b == p]
    for m in range(16):
        np.testing.assert_array_equal(chars[m, diagonal], chars[m ^ 6, diagonal])
    assert np.linalg.matrix_rank(chars[:, diagonal]) == 8
    assert np.linalg.matrix_rank(chars) == 16


@pytest.mark.parametrize("mask", range(16))
def test_single_character_identification_and_omitted_carrier_bounds(mask):
    x = U.characters()[mask].reshape(2, 2, 2, 2)
    coefficients, stats = U.split(x)
    np.testing.assert_array_equal(coefficients, np.eye(16)[mask])
    assert stats[U.LABELS[mask] + "_energy"] == 1
    for name, bit in (("HQ", 6), ("HPQ", 4), ("HBQ", 2)):
        assert stats[name + "_only_mse_bound"] == int(bool(mask & bit))


def test_vector_parseval_conditional_variance_and_old_restriction():
    x = np.random.default_rng(33).normal(size=(2, 2, 2, 2, 3, 4))
    coefficients, stats = U.split(x)
    assert stats["reconstruction_error"] < 1e-14
    for name, axis in (("HQ", (1, 2)), ("HPQ", (1,)), ("HBQ", (2,))):
        assert stats[name + "_only_mse_bound"] == pytest.approx(
            np.sum((x - x.mean(axis=axis, keepdims=True)) ** 2) / 16
        )
    diagonal = np.array(
        [[[x[i, j, j, k] for k in range(2)] for j in range(2)] for i in range(2)]
    )
    np.testing.assert_allclose(
        U.collapse_coefficients(coefficients), C.split(diagonal)[0], atol=1e-14
    )


def test_independent_p_and_b_cannot_be_hidden_on_the_old_diagonal():
    x = (U.characters()[4] - U.characters()[2]).reshape(2, 2, 2, 2)
    c, stats = U.split(x)
    np.testing.assert_array_equal(U.collapse_coefficients(c), np.zeros(8))
    assert stats["HQ_only_mse_bound"] == 2
    assert stats["HPQ_only_mse_bound"] == stats["HBQ_only_mse_bound"] == 1


def test_known_carrier_contrasts_hold_other_three_fixed():
    for mask, name in ((4, "B"), (2, "P")):
        contrasts = U.contrasts(U.characters()[mask].reshape(2, 2, 2, 2))
        assert len(contrasts) == 32
        assert all(
            c["difference_norm"] == (2 if c["vary"] == name else 0) for c in contrasts
        )


def sample_row():
    """Return an explicitly disjoint property/query/endpoint address example."""
    return {
        "input_ids": list(range(10)),
        "question_start": 6,
        "spans": {"A_property": {"positions": [0, 1, 2, 3]}},
    }


def test_positions_keep_endpoint_and_body_separate():
    assert U.positions(sample_row()) == [6, 7, 8]
    with pytest.raises(ValueError):
        U.positions({**sample_row(), "question_start": 3})


def test_mapped_p_noop_and_postpatch_gradient_with_b_fixed():
    original = torch.arange(30.0, requires_grad=True).reshape(1, 10, 3)
    receiver = {
        "h": original[0, :4].detach().numpy(),
        "p": original[0, 6:9].detach().numpy(),
        "q": original[0, -1].detach().numpy(),
    }
    leaf, audit = U.replace_leaf(
        original, [0, 1, 2, 3], [6, 7, 8], receiver, receiver, False
    )
    np.testing.assert_array_equal(leaf.numpy(), original.detach().numpy())
    assert all(v == 0 for v in audit.values())
    donors = {**receiver, "p": receiver["p"] + 3}
    leaf, audit = U.replace_leaf(
        original, [0, 1, 2, 3], [6, 7, 8], receiver, donors, True
    )
    g = torch.autograd.grad((leaf[0, :4] * leaf[0, 6:9].mean(0)).sum(), leaf)[0][0, :4]
    np.testing.assert_array_equal(
        g.numpy(), np.broadcast_to(donors["p"].mean(0), (4, 3))
    )
    np.testing.assert_array_equal(
        leaf.detach().numpy()[0, 4:6], original.detach().numpy()[0, 4:6]
    )
    assert leaf.is_leaf and all(v == 0 for v in audit.values())


def test_changed_receiver_p_and_overlapping_positions_rejected():
    original = torch.zeros(1, 10, 3)
    receiver = {"h": np.zeros((4, 3)), "p": np.zeros((3, 3)), "q": np.zeros(3)}
    with pytest.raises(ValueError):
        U.replace_leaf(original, [0, 1, 2, 3], [3, 6, 7], receiver, receiver, False)
    with pytest.raises(ValueError):
        U.replace_leaf(
            original,
            [0, 1, 2, 3],
            [6, 7, 8],
            {**receiver, "p": np.ones((3, 3))},
            receiver,
            False,
        )


def test_physical_h_delta_independent_of_other_carriers():
    h = np.random.default_rng(3).normal(size=(4, 3)).astype(np.float32)
    d = np.ones_like(h)
    expected = D.physical_patches(h, d)["half_delta"]
    for b, p, q in itertools.product(range(2), repeat=3):
        original = torch.ones(1, 10, 3) * b
        receiver = {
            "h": original[0, :4].numpy(),
            "p": original[0, 6:9].numpy(),
            "q": original[0, -1].numpy(),
        }
        donors = {
            "h": h,
            "p": np.full((3, 3), p, np.float32),
            "q": np.full(3, q, np.float32),
        }
        leaf, _ = U.replace_leaf(
            original, [0, 1, 2, 3], [6, 7, 8], receiver, donors, False
        )
        np.testing.assert_array_equal(
            expected, D.physical_patches(leaf[0, :4].numpy(), d)["half_delta"]
        )


def test_invalid_and_zero_observations():
    assert U.split(np.zeros((2, 2, 2, 2)))[1]["HQ_only_relative_rmse_bound"] is None
    with pytest.raises(ValueError):
        U.split(np.zeros((2, 2, 2, 1)))
    with pytest.raises(ValueError):
        U.split(np.full((2, 2, 2, 2), np.nan))


def test_alternate_product_projections_match_every_character():
    x = np.random.default_rng(39).normal(size=(2, 2, 2, 2, 4, 7))
    coeff, stats = U.split(x)
    alternate, other = product_parts(x)
    np.testing.assert_allclose(coeff, alternate, atol=1e-14)
    for key in stats:
        assert other[key] == pytest.approx(stats[key], abs=1e-12)


def design_fixture():
    """Two natural receivers and a full predecessor cube with exact suffix IDs."""
    natural = [{**sample_row(), "unit": side, "role": "source"} for side in range(2)]
    old = [
        {
            **natural[b],
            "unit": unit,
            "receiver_unit": b,
            "pair": 0,
            "wording": 0,
            "query": "q",
            "donor_side": h,
            "receiver_side": b,
            "query_side": q,
            "inherited": b == q,
        }
        for unit, (h, b, q) in enumerate(itertools.product(range(2), repeat=3))
    ]
    return old, natural


def test_design_has16_corners_eight_inherited_and_bound_donor_addresses():
    old, natural = design_fixture()
    rows = U.design(old, natural)
    assert len(rows) == 16 and sum(r["inherited"] for r in rows) == 8
    assert {
        (r["donor_side"], r["receiver_side"], r["early_query_side"], r["query_side"])
        for r in rows
    } == set(U.BITS)
    for r in rows:
        assert r["early_query_donor_unit"] == r["early_query_side"]
        assert r["cube_unit"] == r["unit"] // 2
        assert r["cube_inherited"] == old[r["cube_unit"]]["inherited"]


@pytest.mark.parametrize("change", ("tokens", "role"))
def test_design_rejects_suffix_or_role_mismatch(change):
    old, natural = design_fixture()
    if change == "tokens":
        natural[1] = {
            **natural[1],
            "input_ids": [
                *natural[1]["input_ids"][:7],
                42,
                *natural[1]["input_ids"][8:],
            ],
        }
    else:
        natural[1] = {**natural[1], "role": "application"}
    with pytest.raises(ValueError):
        U.design(old, natural)
