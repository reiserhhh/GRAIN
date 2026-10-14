#!/usr/bin/env python3
"""HM31 counterexamples, exact interventions and fixed product projections."""

import numpy as np
import pytest
import torch

from grain_research.construction import math_context_fiber as F
from grain_research.construction import math_position_support as P


def test_projection_energy_and_reconstruction():
    x = np.random.default_rng(41).normal(size=(2, 2, 4, 3))
    parts, stats = F.split(x)
    np.testing.assert_allclose(
        x,
        parts["mean"]
        + parts["donor"][:, None]
        + parts["receiver"][None]
        + parts["interaction"],
    )
    assert stats["energy_identity_error"] < 1e-14
    assert sum(
        stats[k + "_share"] for k in ("donor", "receiver", "interaction")
    ) == pytest.approx(1)


def test_context_offset_changes_level_not_action():
    h, c = np.array([1.0, 2.0]), np.array([0.0, 5.0])
    values = h[:, None] ** 2 + c[None]
    gradients = np.broadcast_to(2 * h[:, None], (2, 2))
    assert F.split(values)[1]["receiver_energy"] > 0
    assert all(
        r["difference_norm"] == 0
        for r in F.contrasts(gradients)
        if r["vary"] == "receiver"
    )


def test_identical_h_context_changes_sensitivity():
    h, c, delta = np.array([1.0, 2.0]), np.array([1.0, -1.0]), 0.01
    gradients = np.broadcast_to(c[None], (2, 2))
    actions = ((h[:, None] + delta) * c - (h[:, None] - delta) * c) / 2
    assert all(
        r["difference_norm"] > 0
        for r in F.contrasts(gradients)
        if r["vary"] == "receiver"
    )
    np.testing.assert_allclose(actions, delta * gradients)


def test_interaction_only_and_swap_equivariance():
    x = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert F.split(x)[1]["interaction_share"] == pytest.approx(1)
    assert (
        F.split(x[::-1])[1]["interaction_energy"] == F.split(x)[1]["interaction_energy"]
    )
    assert F.split(x.T)[1]["donor_energy"] == F.split(x)[1]["receiver_energy"]


def test_task_seminorm_can_be_blind():
    x = np.zeros((2, 2, 3))
    x[:, 0, 2] = 1
    assert F.split(x)[1]["receiver_energy"] > 0
    assert F.split(x[..., :2])[1]["total_centered_energy"] == 0


def test_leaf_gradient_after_patch_not_receiver_or_donor_graph():
    original = torch.arange(18.0, requires_grad=True).reshape(1, 6, 3)
    receiver = original[0, :4].detach().numpy()
    donor = receiver.copy() + 2
    leaf, audit = F.replace_leaf(original, [0, 1, 2, 3], receiver, donor, True)
    score = (leaf[0, :4] ** 2).sum() + leaf[0, 5].sum()
    g = torch.autograd.grad(score, leaf)[0][0, :4]
    np.testing.assert_allclose(g.numpy(), 2 * donor)
    assert leaf.is_leaf and all(v == 0 for v in audit.values())
    np.testing.assert_array_equal(
        leaf.detach().numpy()[:, 4:], original.detach().numpy()[:, 4:]
    )


@pytest.mark.parametrize("positions", [[0, 0, 2, 3], [0, 1, 2, 5], [-1, 0, 1, 2]])
def test_bad_address_rejected(positions):
    with pytest.raises(ValueError):
        F.replace_leaf(
            torch.zeros(1, 6, 3), positions, np.zeros((4, 3)), np.zeros((4, 3)), False
        )


def test_receiver_mismatch_rejected():
    with pytest.raises(ValueError):
        F.replace_leaf(
            torch.zeros(1, 6, 3), [0, 1, 2, 3], np.ones((4, 3)), np.zeros((4, 3)), True
        )


def test_identical_donor_gives_identical_physical_delta():
    h = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    a, b = (
        P.physical_patches(h, np.ones((4, 3))),
        P.physical_patches(h.copy(), np.ones((4, 3))),
    )
    np.testing.assert_array_equal(a["half_delta"], b["half_delta"])


def test_corners_and_token_identity():
    rows = [
        {
            "context": c,
            "wording": 0,
            "query": "q",
            "unit": c,
            "role": "source",
            "input_ids": [1, 2, 3, 4, 5],
            "spans": {"A_property": {"positions": [0, 1, 2, 3]}},
        }
        for c in range(2)
    ]
    design = F.corners(rows, [[0, 1]])
    assert len(design) == 4 and sum(r["diagonal"] for r in design) == 2
    rows[1]["input_ids"][0] = 9
    with pytest.raises(ValueError):
        F.corners(rows, [[0, 1]])


def test_incomplete_or_nonfinite_rejected():
    for x in (np.ones((2, 3)), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError):
            F.split(x)
