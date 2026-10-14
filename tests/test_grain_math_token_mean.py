"""Synthetic tests for HM12's fixed decomposition and intervention contract."""

import numpy as np
import pytest

from grain_research.construction import math_token_mean as M
from scripts.run_grain_math_token_mean import family_interval, prepare


def test_full_is_mean_plus_centered():
    rng = np.random.default_rng(12)
    h, donor = rng.normal(size=(5, 17)), rng.normal(size=(5, 17))
    mean, centered = M.mean_centered_displacements(h, donor)
    np.testing.assert_allclose(mean + centered, donor - h, atol=1e-15)
    np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(np.sum(mean * centered), 0, atol=1e-14)


def test_single_token_centered_is_exact_noop():
    h = np.array([[1.25, -2.0, 0.125]], np.float32)
    donor = np.array([[-8.0, 5.0, 2.25]], np.float32)
    np.testing.assert_array_equal(M.patch_target(h, donor, "centered_only"), h)
    np.testing.assert_array_equal(M.patch_target(h, donor, "mean_only"), donor)
    assert M.geometry_audit(h, donor, "centered_only")["is_exact_input_noop"]


def test_centered_patch_preserves_full_dimensional_mean_before_rounding():
    h = np.array([[0.5, 1], [2, -1]], np.float32)
    donor = np.array([[1, 3], [4, 0]], np.float32)
    patched = M.patch_target(h, donor, "centered_only")
    np.testing.assert_array_equal(patched.mean(axis=0), h.mean(axis=0))
    assert M.geometry_audit(h, donor, "centered_only")["intervention_norm"] > 0


@pytest.mark.parametrize(
    "h,d",
    [
        (np.zeros(2), np.zeros(2)),
        (np.zeros((0, 3)), np.zeros((0, 3))),
        (np.zeros((2, 3)), np.zeros((3, 3))),
        (np.array([[np.nan]]), np.zeros((1, 1))),
    ],
)
def test_bad_token_states_fail(h, d):
    with pytest.raises(ValueError):
        M.mean_centered_displacements(h, d)


def test_unknown_arm_fails():
    with pytest.raises(ValueError):
        M.patch_target(np.zeros((2, 3)), np.ones((2, 3)), "rank_grid")


def test_finite_mixed_response_is_retained():
    row = M.finite_response_split(0.1, 0.7, 0.2, 0.3)
    assert row["mixed_action"] == pytest.approx(0.3)
    assert row["full_action"] == pytest.approx(
        row["mean_action"] + row["centered_action"] + row["mixed_action"]
    )


def test_design_uses_all_queries_and_aligned_spans():
    spec = {
        "layer": 16,
        "arms": ["mean_only", "centered_only"],
        "queries": ["q1", "q2", "q3"],
    }
    pairs = [{"pair_id": 0, "axis": "A", "receiver_doc": 0, "donor_doc": 1}]
    span = {"positions": [2, 3], "token_ids": [8, 9], "raw": "same words"}
    inputs = {
        (d, q): {"question_start": 9, "spans": {"A_property": dict(span)}}
        for d in (0, 1)
        for q in spec["queries"]
    }
    rows = M.enumerate_interventions(pairs, inputs, spec)
    assert len(rows) == 6
    assert {r["unit"] for r in rows} == set(range(6))
    assert all(r["opportunity"] == "multi" for r in rows)
    inputs[1, "q3"]["spans"]["A_property"]["raw"] = "different words"
    with pytest.raises(ValueError):
        M.enumerate_interventions(pairs, inputs, spec)


def test_family_intervals_are_reproducible():
    spec = {"bootstrap_seed": 12, "bootstrap_replicates": 500}
    assert family_interval([1, 2, 3, 4], spec) == family_interval([1, 2, 3, 4], spec)
    assert family_interval([5], spec) == (5, 5, 5)


def test_prepare_refuses_existing_packet(tmp_path):
    with pytest.raises(FileExistsError):
        prepare(tmp_path)
