#!/usr/bin/env python3
"""Minimal mechanism and cold-input tests, independent of the local model."""

import numpy as np
import pytest

from grain_research.construction import math_writer_history_bridge as M


def test_current_and_historical_tokenizer_formats():
    assert M.rendered_token_ids([3, 5]) == [3, 5]
    assert M.rendered_token_ids({"input_ids": [3, 5], "attention_mask": [1, 1]}) == [
        3,
        5,
    ]
    with pytest.raises(ValueError):
        M.rendered_token_ids({"input_ids": [[3, 5]]})


def test_order_keeps_semantic_choice_and_exposes_option_mass():
    a = M.choice_account([2.0, -1.0], 3.0, 0)
    b = M.choice_account([-1.0, 2.0], 3.0, 1)
    assert a == b
    assert a["q_positive"] > 0.9
    assert a["unrestricted_option_mass"] < 0.4


def test_account_signed_components_not_information_fractions():
    q = np.array([[0.6, 0.2, 0.1], [0.4, 0.3, 0.7], [0.2, 0.5, 0.8]])
    a = M.nine_corner_account(q)
    assert a["reconstruction_error"] < 1e-15
    assert a["natural_effect"] == pytest.approx(
        a["policy_effect"] + a["history_effect"] + a["interaction_effect"]
    )
    assert a["interaction_effect"] < 0


@pytest.mark.parametrize("driver", ["policy", "history", "constant"])
def test_competing_null_laws_are_distinguishable(driver):
    levels = np.arange(3.0) / 4
    q = {
        "policy": np.repeat(levels[:, None], 3, axis=1),
        "history": np.repeat(levels[None, :], 3, axis=0),
        "constant": np.full((3, 3), 0.3),
    }[driver]
    a = M.nine_corner_account(q)
    assert a["policy_effect"] == pytest.approx(0.5 if driver == "policy" else 0)
    assert a["history_effect"] == pytest.approx(0.5 if driver == "history" else 0)
    assert a["interaction_effect"] == pytest.approx(0)


def test_requests_cross_real_bodies_and_keep_old_task_and_style():
    pair = {
        "pair_id": "pair",
        "axis": "e",
        "situation": "diary",
        "style": "plain",
        "seed": 11,
        "role": "discovery",
        "old_user": "old task",
    }
    opportunity = {
        "id": "new",
        "domain": "d",
        "condition": "c",
        "context": "context",
        "positive": "speak",
        "negative": "leave",
    }
    rows = M.bridge_requests(
        pair,
        {-1: "low", 0: "neutral", 1: "high"},
        {-1: "actual low", 0: "", 1: "actual high"},
        [opportunity],
    )
    assert len(rows) == 18
    assert len({r["request_id"] for r in rows}) == 18
    for r in rows:
        assert r["messages"][1]["content"] == "old task"
        assert (
            r["messages"][2]["content"]
            == {-1: "actual low", 0: "", 1: "actual high"}[r["history"]]
        )
        assert "judging another person" in r["messages"][3]["content"]


def test_source_only_witness_and_independent_pairwise_formula():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(12, 5)), rng.uniform(size=(12, 4))
    origin, c, norms = M.response_witness(x, y)
    pairwise = np.einsum(
        "ijc,ijd->cd", y[:, None] - y[None, :], x[:, None] - x[None, :]
    ) / (2 * len(x) ** 2)
    np.testing.assert_allclose(c, pairwise, atol=1e-15)
    result = M.cold_response_reading(x[:2], origin, c, norms)
    assert result.shape == (2, 4)
    with pytest.raises(ValueError):
        M.cold_response_reading(x[:, :4], origin, c, norms)


def test_constant_response_unavailable_not_successful_zero_score():
    origin, c, norms = M.response_witness(
        np.arange(12.0).reshape(4, 3), np.ones((4, 2))
    )
    assert np.isnan(M.cold_response_reading(np.ones((2, 3)), origin, c, norms)).all()


def test_one_draw_per_registered_event_is_reproducible():
    assert M.sample_protocol_choice(0.7, "fixed", 165) == M.sample_protocol_choice(
        0.7, "fixed", 165
    )
    assert M.sample_protocol_choice(0, "fixed", 165) == 0
    assert M.sample_protocol_choice(1, "fixed", 165) == 1


def test_bad_support_is_rejected():
    with pytest.raises(ValueError):
        M.nine_corner_account(np.ones((2, 2)))
    with pytest.raises(ValueError):
        M.choice_account([0, 0], -1, 0)
