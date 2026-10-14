#!/usr/bin/env python3
"""Mechanism/null identities and fixed-sign controls; no neural calls."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_evidence_response_operation as M
from grain_research.construction.math_relational_cut import call_allowed
from scripts.verify_grain_math_evidence_response_operation import (
    null_scalar,
    pair_scalar,
    summary_scalar,
)


@pytest.mark.parametrize("offset", [-20, -2, 0, 2, 20])
def test_output_constant_offset_has_zero_log_interaction(offset):
    logits = [np.arange(10, dtype=float) * sign for sign in (0.2, -0.7)]
    n = [M.output_offset(z, 0) for z in logits]
    p = [M.output_offset(z, offset) for z in logits]
    row = M.paired(*n, *p)
    assert row["mixed_log_odds"] == pytest.approx(0, abs=1e-12)
    for z, v in zip(logits, p, strict=True):
        q = v["probabilities"]
        assert q[8] / q[7] == pytest.approx(np.exp(z[8] - z[7]))
        assert q[2] / q[1] == pytest.approx(np.exp(z[2] - z[1]))


def test_score_saturation_can_make_interaction_without_changed_log_response():
    z0, z1 = np.zeros(10), np.r_[np.zeros(5), np.ones(5) * 5]
    n = [M.output_offset(z, 0) for z in (z0, z1)]
    p = [M.output_offset(z, 3) for z in (z0, z1)]
    row = M.paired(*n, *p)
    assert abs(row["mixed_score"]) > 0.1
    assert abs(row["mixed_log_odds"]) < 1e-12


def test_nonlinear_payload_has_mixed_difference_without_reading_program():
    # A fixed additive payload passed through a square is a counterexample.
    n = [{"score": 0, "log_odds": x * x} for x in (0, 2)]
    p = [{"score": 0, "log_odds": (x + 1) ** 2} for x in (0, 2)]
    assert M.paired(*n, *p)["mixed_log_odds"] == 4


@pytest.mark.parametrize("arm", M.ARMS)
def test_sign_and_random_are_independent_of_receiver_direction(arm):
    current, donor = np.zeros((6, 3), np.float32), np.ones((6, 3), np.float32)
    source = {"e__A": np.ones((4, 3)), "e__B": np.ones((4, 3)) * 2}
    r = {
        "content": "e",
        "person": "A",
        "pair_id": 3,
        "partition": {"suffix": [2, 3, 4, 5]},
    }
    e = {"arm": arm, "donor_positions": [2, 3, 4, 5]}
    values = [
        M.patch_values(e, r | {"direction": d}, current, donor, source, 5401)[1]
        for d in (0, 1)
    ]
    np.testing.assert_array_equal(*values)
    expected = {
        "source_plus": 1,
        "source_minus": -1,
        "wrong_person": 2,
        "exact_suffix": 1,
        "noop": 0,
    }
    if arm in expected:
        np.testing.assert_array_equal(values[0], expected[arm])
    else:
        np.testing.assert_allclose(np.linalg.norm(values[0], axis=1), np.sqrt(3))


def test_full_materials_preserve_ownership_bags_and_scope():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "configs/grain_math_evidence_response_operation_v1.json").read_text()
    )
    docs = M.documents(spec)
    assert len(docs) == 48
    assert {r["class"] for r in docs} == set(M.CLASSES)
    assert len({r["body"] for r in docs}) == 38
    assert all(
        len(spec["fresh_behaviours"][t][f]) == 2
        for t in spec["targets"]
        for f in ("family_1", "family_2")
    )


@pytest.mark.parametrize(
    "logits,offset",
    [(np.zeros(9), 1), (np.ones(10) * np.nan, 0), (np.ones(10), np.inf)],
)
def test_invalid_offset_inputs_fail(logits, offset):
    with pytest.raises(ValueError):
        M.output_offset(logits, offset)


def test_summaries_keep_signed_interactions():
    a = {"score": 0.1, "log_odds": 1}
    b = {"score": 0.6, "log_odds": 2}
    p = {"score": 0.3, "log_odds": 4}
    q = {"score": 0.2, "log_odds": 3}
    row = M.paired(a, b, p, q)
    summary = M.paired_summary([row])
    assert summary["mixed_log_odds_mean"] == -2
    assert summary["mixed_log_odds_rms"] == 2
    assert summary["native_patched_log_direction_matches"] == 0


def test_bounded_call_guard():
    spec = {"max_forwards": 672, "max_model_seconds": 600}
    status = {"status": "PREPARED", "attempted_forward_calls": 0, "in_flight": None}
    assert call_allowed(status, spec, 0)
    assert not call_allowed(status | {"attempted_forward_calls": 672}, spec, 0)
    assert not call_allowed(status | {"in_flight": 1}, spec, 0)
    assert not call_allowed(status, spec, 600)


@pytest.mark.parametrize("offset", [-12, 0, 9])
def test_alternate_arithmetic_reconstructs_null_and_pair(offset):
    logits = [np.linspace(-3, 2, 10), np.linspace(1, -1, 10)]
    native = [M.output_offset(z, 0) for z in logits]
    patched = [M.output_offset(z, offset) for z in logits]
    alternate = [null_scalar(z, offset) for z in logits]
    for actual, expected in zip(alternate, patched, strict=True):
        assert actual["score"] == pytest.approx(expected["score"], abs=1e-12)
        assert actual["log_odds"] == pytest.approx(expected["log_odds"], abs=1e-12)
        np.testing.assert_allclose(actual["probabilities"], expected["probabilities"])
    actual, expected = pair_scalar(*native, *patched), M.paired(*native, *patched)
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)
    for key, value in summary_scalar([actual]).items():
        assert value == pytest.approx(M.paired_summary([expected])[key], abs=1e-12)
