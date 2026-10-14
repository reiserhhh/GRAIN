#!/usr/bin/env python3
"""Mechanism checks for actual crossed outputs, not prediction-accuracy gates."""

import numpy as np
import pytest

from grain_research.construction import math_joint_persona_style as M
from scripts.run_grain_semantic_anchor_hm147 import SOURCE, load


@pytest.fixture
def rows():
    return M.joint_materials(load(SOURCE / "manifest.json"))


def test_complete_all_original_factors(rows):
    assert len(rows) == len({r["text_id"] for r in rows}) == 702
    batches = {r["batch"] for r in rows}
    assert len(batches) == 54
    assert all(sum(r["batch"] == b for r in rows) == 13 for b in batches)
    assert sum(r["axis"] == "none" for r in rows) == 54
    assert {r["family"] for r in rows if r["pole"]} == set(M.FAMILIES)
    assert {r["style"] for r in rows} == set(M.STYLES)


def test_joint_not_replacement_prompt(rows):
    original = load(SOURCE / "manifest.json")
    for row in rows:
        assert row["system"].endswith(original["shared_suffix"])
        if row["style"] != "neutral":
            assert original["nonpersona"][row["style"]] in row["system"]
        if row["pole"]:
            pole = "high" if row["pole"] == 1 else "low"
            assert (
                original["aligned_pairs"][f"{row['axis']}:{row['family']}"][pole]
                in row["system"]
            )


def test_actual_corner_support_and_additive_null(rows):
    corners, meta = M.interaction_design(rows)
    assert corners.shape == (216, 4)
    assert len(meta) == 216
    x = np.array([[r["pole"], M.STYLES.index(r["style"])] for r in rows], float)
    np.testing.assert_allclose(M.mixed_difference(x, corners), 0)


def test_real_modulation_and_roles(rows):
    corners, meta = M.interaction_design(rows)
    x = np.array([[r["pole"] * M.STYLES.index(r["style"])] for r in rows], float)
    expected = np.array([[2 * M.STYLES.index(m["style"])] for m in meta])
    np.testing.assert_allclose(M.mixed_difference(x, corners), expected)
    assert sum(m["role"] == "evaluation" for m in meta) == 72


def test_missing_actual_corner_rejected(rows):
    with pytest.raises(ValueError, match="Missing actual"):
        M.interaction_design(rows[1:])


def test_duplicate_actual_corner_rejected(rows):
    with pytest.raises(ValueError, match="Duplicate"):
        M.interaction_design(rows + rows[:1])


def test_nonfinite_values_rejected():
    with pytest.raises(ValueError, match="Finite"):
        M.mixed_difference([[np.nan], [1]], [[0, 1, 0, 1]])


def test_feature_curvature_can_exist_without_input_interaction():
    z = np.zeros((4, 4, 4))
    z[1, 0, 0] = 2
    z[2, 1, 0] = 3
    z[0] = z[1] + z[2] - z[3]
    corner = np.array([[0, 1, 2, 3]])
    np.testing.assert_allclose(M.mixed_difference(z, corner), 0)
    parts = M.tensor_accounts(z, corner)
    assert np.linalg.norm(parts["observed"]) > 0
    np.testing.assert_allclose(parts["upper_action"], 0)
    np.testing.assert_allclose(parts["lower_action"], 0)


def test_both_anchor_accounts_reconstruct():
    z = np.random.default_rng(5).normal(size=(8, 4, 4))
    parts = M.tensor_accounts(z, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    for prefix in ("upper", "lower"):
        np.testing.assert_allclose(
            parts["observed"],
            parts[f"{prefix}_link"] + parts[f"{prefix}_action"],
            atol=1e-12,
        )


def test_negative_cross_seed_energy_not_clipped():
    s = M.seed_support([[1, 0], [-1, 0], [0, 0]])
    assert s["cross_seed_support"] == pytest.approx(-1 / 3)
    assert s["mean_squared_norm"] == 0
    assert s["identity_error"] < 1e-12


def test_repeatable_response_positive():
    s = M.seed_support([[1, 2]] * 3)
    assert s["cross_seed_support"] == 5
    assert s["within_seed_term"] == 0


def test_three_seed_requirement():
    with pytest.raises(ValueError, match="Exactly three"):
        M.seed_support([[1], [1]])
