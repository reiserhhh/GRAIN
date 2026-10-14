"""Synthetic tests for exact HM13 selection, multiset action and no-op controls."""

import numpy as np
import pytest

from grain_research.construction import math_token_permutation as P
from grain_research.construction.math_token_mean import patch_target


def test_reverse_twice_and_byte_multiset():
    h = np.array([[1, -0.0, 3], [4, 5, 6], [7, 8, 9]], np.float32)
    ref, target = P.reference_and_target(h, h + 2, "receiver_reverse")
    assert ref.tobytes() == h.tobytes()
    assert target[::-1].copy().tobytes() == h.tobytes()
    assert P.row_bytes(ref) == P.row_bytes(target)
    audit = P.permutation_audit(h, h + 2, "receiver_reverse")
    assert audit["mean_replay_max"] == 0
    assert audit["covariance_replay_max"] < 1e-14
    assert not audit["is_exact_input_noop"]


def test_centered_assignment_is_exact_frozen_float32_formula():
    rng = np.random.default_rng(13)
    h, d = (rng.normal(size=(4, 17)).astype(np.float32) for _ in range(2))
    reference, target = P.reference_and_target(h, d, "centered_reverse")
    assert reference.tobytes() == patch_target(h, d, "centered_only").tobytes()
    assert target.tobytes() == reference[::-1].copy().tobytes()
    assert P.permutation_audit(h, d, "centered_reverse")["row_multiset_exact"]


@pytest.mark.parametrize("arm", ["receiver_reverse", "centered_reverse"])
def test_single_token_is_exact_actual_noop(arm):
    h = np.array([[1.25, -2, 0.125]], np.float32)
    d = np.array([[-4, 9, 5]], np.float32)
    audit = P.permutation_audit(h, d, arm)
    assert audit["is_exact_input_noop"]
    assert audit["permutation_is_identity"]
    assert audit["total_intervention_norm"] == 0


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 3), float),
        np.zeros((0, 3), np.float32),
        np.zeros(3, np.float32),
        np.array([[np.nan]], np.float32),
    ],
)
def test_invalid_token_states_fail(value):
    with pytest.raises(ValueError):
        P.token_matrix(value)


def test_unknown_arm_and_mismatched_shapes_fail():
    h = np.zeros((2, 3), np.float32)
    with pytest.raises(ValueError):
        P.reference_and_target(h, h, "prompt_grid")
    with pytest.raises(ValueError):
        P.reference_and_target(h, h[:1], "receiver_reverse")


def test_signed_zero_is_not_silently_dropped_from_certificate():
    plus = np.array([[0.0]], np.float32)
    minus = np.array([[-0.0]], np.float32)
    assert P.row_bytes(plus) != P.row_bytes(minus)


def test_endpoint_bound_and_nonlinear_position_counterexample():
    h = np.array([[1], [3]], np.float32)
    _, target = P.reference_and_target(h, h, "receiver_reverse")
    readout = lambda x: float(x[0, 0] * x[1, 0] ** 2)
    result = P.endpoint_effect(readout(target), readout(h))
    assert result["effect"] == -6
    assert result["set_function_max_error_lower_bound"] == 3
    assert h.mean() == target.mean()
    with pytest.raises(ValueError):
        P.endpoint_effect(np.nan, 0)


def small_design():
    """Make a six-cell source-independent fixture with two possible no-op IDs per cell."""
    spec = {
        "layer": 16,
        "arms": ["receiver_reverse", "centered_reverse"],
        "roles": ["source", "validation", "application"],
        "axes": ["A", "T"],
        "queries": ["q1", "q2", "q3"],
        "original_pairs": 13,
        "multi_pairs": 1,
        "patch_forwards": 42,
    }
    pairs = [{"pair_id": 0, "role": "source", "axis": "A", "tokens": 2}]
    for role in spec["roles"]:
        for axis in spec["axes"]:
            for _ in range(2):
                pairs.append(
                    {"pair_id": len(pairs), "role": role, "axis": axis, "tokens": 1}
                )
    rows = []
    for pair in pairs:
        for q in spec["queries"]:
            for arm in ("mean_only", "centered_only"):
                rows.append(
                    {
                        **pair,
                        "query": q,
                        "arm": arm,
                        "unit": len(rows),
                        "positions": list(range(pair["tokens"])),
                        "token_ids": [8] * pair["tokens"],
                    }
                )
    return rows, spec


def test_design_selects_all_multi_and_smallest_controls_without_output_scores():
    rows, spec = small_design()
    design = P.enumerate_interventions(list(reversed(rows)), spec)
    assert len(design) == 42
    assert {r["pair_id"] for r in design} == {0, 1, 3, 5, 7, 9, 11}
    assert [r["unit"] for r in design] == list(range(42))
    assert {r["arm"] for r in design} == set(spec["arms"])


def test_design_rejects_missing_query_and_inconsistent_opportunity():
    rows, spec = small_design()
    rows[2]["positions"] = [99, 100]
    with pytest.raises(ValueError):
        P.enumerate_interventions(rows, spec)


def test_prepare_preserves_existing_packet(tmp_path):
    from scripts.run_grain_math_token_permutation import prepare

    with pytest.raises(FileExistsError):
        prepare(tmp_path)
