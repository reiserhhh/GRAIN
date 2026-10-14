#!/usr/bin/env python3
"""Synthetic mechanism and scope checks before costly finite acquisitions."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_causal_ownership_sectors as M

SPEC = (
    Path(__file__).resolve().parents[1]
    / "configs/grain_causal_ownership_sectors_hm74.json"
)


def test_complete_new_material_scope():
    """All targets and directions are represented independently of readout outcomes."""
    documents = M.documents(json.loads(SPEC.read_text()))
    assert len(documents) == 288
    assert len({r["body"] for r in documents}) == 288
    assert len({(r["focal"], r["companion"]) for r in documents}) == 36
    assert sum(1 if r["companion"] == "neutral" else 2 for r in documents) == 528
    assert {r["focal"] for r in documents} == set(M.TARGETS)


def test_lossless_character_expansion():
    """Full finite reconstruction is exact without discarding reference level."""
    states = np.random.default_rng(74).normal(size=(8, 4, 6)).astype(np.float32)
    assert np.max(abs(M.reconstruct(M.coefficients(states)) - states)) < 1e-12


@pytest.mark.parametrize(
    "arm,removed",
    [
        ("remove_yg", (3,)),
        ("remove_xyg", (7,)),
        ("remove_both", (3, 7)),
        ("remove_xy", (6,)),
    ],
)
def test_only_declared_coefficients_removed(arm, removed):
    """Constant, main effects and other interactions survive numerical assignments."""
    states = np.random.default_rng(4).normal(size=(8, 3, 5)).astype(np.float32)
    before, after = M.coefficients(states), M.coefficients(M.payload(states, arm, 1))
    for j in range(8):
        assert np.max(abs(after[j] - (0 if j in removed else before[j]))) < 2e-7


def test_linear_storage_and_nonlinear_recomputation_are_distinct():
    """Output interactions need not reside in the same hidden-state characters."""
    linear, nonlinear = M.toy_world("linear_storage"), M.toy_world("recomputed")
    assert np.array_equal(linear["noop"], nonlinear["noop"])
    assert np.array_equal(linear["remove_yg"], np.zeros(8))
    assert np.array_equal(nonlinear["remove_yg"], nonlinear["noop"])
    assert M.coefficients(nonlinear["remove_both"])[3] == 1


def test_context_dependent_effect_can_cancel_in_average():
    """A zero marginal action does not erase conditional opposite actions."""
    values = M.toy_world("conditional")
    assert np.array_equal(M.ownership_actions(values["noop"])[:, 0], [0, -4, 4])
    assert np.array_equal(values["remove_xyg"], np.zeros(8))


def test_norm_matched_random_preserves_parity_and_seed():
    """Random control has matched intervention norm without output-dependent choice."""
    states = np.random.default_rng(3).normal(size=(8, 4, 9)).astype(np.float32)
    raw = states - M.payload(states, "remove_both", 17)
    random = states - M.payload(states, "random_both", 17)
    assert (
        np.max(abs(np.linalg.norm(raw, axis=-1) - np.linalg.norm(random, axis=-1)))
        < 5e-7
    )
    assert np.array_equal(random, states - M.payload(states, "random_both", 17))
    assert not np.array_equal(random, states - M.payload(states, "random_both", 18))
    coefficients = M.coefficients(random)
    assert np.max(abs(coefficients[[0, 1, 2, 4, 5, 6]])) < 2e-7


def test_donor_and_noop_exact_assignments():
    """Positive and null controls do not interpolate suffix positions."""
    states = np.random.default_rng(2).normal(size=(8, 4, 3)).astype(np.float32)
    assert np.array_equal(M.payload(states, "noop", 0), states)
    assert np.array_equal(M.payload(states, "exact_owner", 0), states[np.arange(8) ^ 1])


def test_author_compositions_and_marginals():
    """Evidence populations can share lower marginals while their observations differ."""
    values = M.SIGNS[7] + 0.3 * M.SIGNS[3]
    result = M.composition_contrasts(values)
    assert result["xyg"]["difference"] == pytest.approx(1)
    assert result["yg"]["difference"] == pytest.approx(0.3)
    plus, minus = result["xyg"]["plus_probability"], result["xyg"]["minus_probability"]
    assert plus.sum() == pytest.approx(1)
    assert np.min(plus) > 0
    assert np.max(abs(M.SIGNS[:7] @ (plus - minus))) < 1e-12


def test_joint_loss_account_exact():
    """Ablation interactions are retained instead of claiming additive mediation."""
    a, b, c, d = np.random.default_rng(9).normal(size=(4, 8))
    interaction = M.joint_ablation(a, b, c, d)
    assert np.max(abs((a - d) - ((a - b) + (a - c) - interaction))) < 1e-12


@pytest.mark.parametrize(
    "states,arm",
    [
        (np.zeros((7, 2, 3), np.float32), "noop"),
        (np.zeros((8, 2, 3)), "noop"),
        (np.zeros((8, 2, 3), np.float32), "unplanned"),
    ],
)
def test_invalid_assignments_fail_closed(states, arm):
    """Incomplete observations and unplanned arms cannot silently be analyzed."""
    with pytest.raises(ValueError):
        M.payload(states, arm, 1)
