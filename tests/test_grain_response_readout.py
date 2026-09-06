"""Synthetic invariance and counterexample tests for exact readout geometry."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.response_readout import (
    decompose_logits,
    state_geometry,
    summarize,
)


def test_tail_head_alignment_with_small_causal_model():
    from types import SimpleNamespace

    import torch
    from test_grain_response_depth import TinyLM

    from grain_research.vector_mechanisms.response_context import ResponseObserver
    from grain_research.vector_mechanisms.response_readout import pair_readout

    torch.manual_seed(731)
    torch.set_num_threads(2)
    model = TinyLM().eval()
    observer = ResponseObserver(
        model, SimpleNamespace(pad_token_id=0), capture_sites=("block_10", "final_norm")
    )
    body = [11, 12, 13, 14]
    va, vm = observer.forward([2, 3], body), observer.forward([4, 5], body)
    rows, checks = pair_readout(model, va, vm, body)
    assert [r["response_token_position"] for r in rows] == [2, 3, 4]
    assert checks["head_replay_max_error"] < 1e-5
    va["trace"]["final_norm"]["states"] = np.roll(
        va["trace"]["final_norm"]["states"], 1, axis=0
    )
    with pytest.raises(ValueError, match="replay"):
        pair_readout(model, va, vm, body)


def test_exact_kl_and_concavity_with_extreme_logits():
    rng = np.random.default_rng(731)
    for scale in (0.01, 1, 50, 100):
        a, m = rng.normal(size=(2, 12, 37)) * scale
        y = rng.integers(37, size=12)
        r, error = decompose_logits(a, m, y)
        assert error < 1e-9
        assert np.all(r["actual_gradient_dot_delta"] <= r["delta_logp"] + 1e-9)
        assert np.all(r["matched_gradient_dot_delta"] >= r["delta_logp"] - 1e-9)


def test_identity_and_common_logit_shift_are_probability_invariant():
    m = np.random.default_rng(3).normal(size=(4, 23))
    for shift in (0, 13, -81):
        r, _ = decompose_logits(m + shift, m, [1, 2, 3, 4])
        for k in (
            "delta_logp",
            "matched_gradient_dot_delta",
            "actual_gradient_dot_delta",
            "kl_matched_to_actual",
            "kl_actual_to_matched",
            "matched_directional_curvature",
        ):
            assert np.max(abs(r[k])) < 1e-9


def test_coordinate_rotation_and_transport_leave_readout_unchanged():
    rng = np.random.default_rng(12)
    a, m = rng.normal(size=(2, 7, 5))
    w = rng.normal(size=(19, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    r1, _ = decompose_logits(a @ w.T, m @ w.T, np.arange(7))
    r2, _ = decompose_logits((a @ q) @ (w @ q).T, (m @ q) @ (w @ q).T, np.arange(7))
    for k in r1:
        np.testing.assert_allclose(r1[k], r2[k], atol=1e-12)


def test_equal_hidden_distances_can_have_opposite_or_zero_target_effects():
    m = np.array([[1.0, 1, 1]] * 3)
    a = m + np.array([[0.1, 0, 0], [-0.1, 0, 0], [0, 0, 0.1]])
    w = np.array([[3, 0, 0], [-3, 0, 0]])
    r, _ = decompose_logits(a @ w.T, m @ w.T, [0, 0, 0])
    geometry, _ = state_geometry(a, m)
    np.testing.assert_allclose(geometry["state_displacement_norm"], 0.1)
    assert r["delta_logp"][0] > 0 > r["delta_logp"][1]
    assert r["delta_logp"][2] == r["kl_matched_to_actual"][2] == 0


@pytest.mark.parametrize("targets", [[1.2, 0], [1], [0, -1], [0, 3]])
def test_bad_targets_rejected(targets):
    with pytest.raises(ValueError):
        decompose_logits(np.zeros((2, 3)), np.zeros((2, 3)), targets)


def test_author_aggregation_does_not_overweight_long_replies():
    f = pd.DataFrame(
        [{"participant_code": "A", "response_token_position": 2, "delta_logp": 1}]
        + [
            {"participant_code": "B", "response_token_position": i + 2, "delta_logp": 0}
            for i in range(99)
        ]
    )
    persons, summary = summarize(
        f,
        [
            {"participant_code": c, "mean_predictor_state_cosine": 0.99}
            for c in ("A", "B")
        ],
    )
    assert len(persons) == 2 and summary["delta_logp"]["mean"] == 0.5
