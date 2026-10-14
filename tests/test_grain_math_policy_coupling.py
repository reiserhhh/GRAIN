#!/usr/bin/env python3
"""Mechanism nulls and typed-policy preservation, not model-accuracy tests."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_policy_coupling as M


def inputs():
    """Provide complete numeric kernels and one counterbalanced eligible ledger."""
    kernel = np.zeros((2, 5, 24, 2, 6, 4))
    joint = M.route_counts(np.full(24, 8))[None]
    return kernel, joint, np.array([0]), np.array([0]), np.array(["instrument"])


def test_recipes_and_choice_keep_conditional_behavior():
    laws = M.recipes().reshape(6, 6, 2, 2)
    np.testing.assert_allclose(laws.sum(axis=(1, 2, 3)), 1)
    assert np.min(laws) > 0
    conditional = laws / laws.sum(axis=3, keepdims=True)
    np.testing.assert_allclose(conditional[4], conditional[5])
    assert not np.allclose(laws[4], laws[5])


def test_coupling_keeps_complete_evidence_and_group_focus_margins():
    n = np.arange(2, 50, 2)
    joint = M.route_counts(n)
    np.testing.assert_array_equal(joint.sum(axis=2), np.broadcast_to(n, (5, 24)))
    grouped = joint.reshape(5, 12, 2, 2).sum(axis=2)
    np.testing.assert_array_equal(grouped[3], grouped[2])
    np.testing.assert_array_equal(grouped[4], grouped[2])


def test_additive_kernel_has_zero_same_margin_effect():
    values = inputs()
    values[0][:] = np.arange(24)[None, None, :, None, None, None]
    values[0][:, :, :, 1] += 7
    out = M.score(*values)
    np.testing.assert_allclose(out["observed"][:, 3], out["observed"][:, 4])


def test_interaction_kernel_and_four_corners_are_observable():
    values = inputs()
    b = M.ATOMS[:, 2]
    values[0][:] = (
        b[None, None, :, None, None, None]
        * np.array([-1, 1])[None, None, None, :, None, None]
    )
    out = M.score(*values)
    difference = out["observed"][0, 3] - out["observed"][0, 4]
    np.testing.assert_allclose(difference, 1)
    np.testing.assert_allclose(difference, M.direct_pair(values[0][0], np.full(24, 8)))


def test_provenance_changes_interpretation_not_numeric_scores():
    values = inputs()
    values[0][:, :, :, 1] = 1
    instrument = M.score(*values)
    author = M.score(*values[:-1], np.array(["author"]))
    np.testing.assert_array_equal(instrument["observed"], author["observed"])
    np.testing.assert_array_equal(instrument["instrument"], author["author_choice"])
    np.testing.assert_array_equal(author["instrument"], 0)
    np.testing.assert_allclose(
        author["observed"], author["reference"] + author["author_choice"]
    )


def test_missing_group_is_missing_not_zero():
    values = inputs()
    values[1][:, :, :2] = 0
    out = M.score(*values)
    assert np.isnan(out["conditional_mean"][:, :, 0]).all()
    assert np.isfinite(out["conditional_mean"][:, :, 1:]).all()


@pytest.mark.parametrize(
    "bad", ["owner", "origin", "fractional", "negative", "empty", "nan"]
)
def test_invalid_ledger_is_rejected(bad):
    values = list(inputs())
    if bad == "owner":
        values[3] = np.array([1])
    elif bad == "origin":
        values[4] = np.array(["unknown"])
    elif bad == "fractional":
        values[1] = values[1].astype(float)
    elif bad == "negative":
        values[1][0, 0, 0, 0] = -1
    elif bad == "empty":
        values[1][:] = 0
    else:
        values[0][0, 0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        M.score(*values)


def test_policy_construction_needs_even_counts():
    with pytest.raises(ValueError):
        M.route_counts(np.full(24, 3))


def test_cold_cli_preserves_joint_policy_without_neural_runtime(tmp_path):
    values = inputs()
    values[0][:] = (
        M.ATOMS[:, 2][None, None, :, None, None, None]
        * np.array([-1, 1])[None, None, None, :, None, None]
    )
    np.savez(tmp_path / "kernels.npz", kernels=values[0])
    np.savez(
        tmp_path / "ledgers.npz",
        joint_counts=values[1],
        evidence_owner=values[2],
        query_owner=values[3],
        origins=values[4],
    )
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "scripts/score_grain_policy_coupling_cold_hm118.py"
        ),
        "--kernels",
        str(tmp_path / "kernels.npz"),
        "--ledgers",
        str(tmp_path / "ledgers.npz"),
        "--output",
        str(tmp_path / "scored.npz"),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with np.load(tmp_path / "scored.npz", allow_pickle=False) as result:
        assert not bool(result["neural_runtime_loaded"])
        np.testing.assert_allclose(
            result["observed"][0, 3] - result["observed"][0, 4], 1
        )
