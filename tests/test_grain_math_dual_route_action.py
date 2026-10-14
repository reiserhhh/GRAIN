#!/usr/bin/env python3
"""Fixed observable, graph completeness, gauge, units and accounting witnesses."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grain_research.construction import math_dual_route_action as D
from grain_research.construction import math_mixed_query_path as MIX


def graph(bypass=False):
    """Synthetic nonlinear sequential graph with residual paths."""
    x = torch.arange(21, dtype=torch.float64).reshape(1, 7, 3).div(30).requires_grad_()
    s = x + torch.sin(x + x.flip(1)) + 0.2 * x.mean(1, keepdim=True)
    logits = torch.stack([s.square().sum(), (s[:, 0] * s[:, -1]).sum()])
    if bypass:
        logits = logits + x[:, 0].sum()
    masks = np.zeros((3, 7), dtype=bool)
    masks[0, :3], masks[1, [3, 5]] = True, True
    masks[2] = ~masks[:2].any(0)
    return x, s, logits, masks


def test_primal_dual_residual_chain_and_call_count():
    x, s, logits, masks = graph()
    w = np.array([0.31, -0.73])
    calls = []
    cov, downstream = D.route_covectors(
        logits, s, x, [0, 2], masks, w, lambda *a: calls.append(a)
    )
    direct = torch.autograd.grad(
        (logits * torch.tensor(w)).sum(), x, retain_graph=True
    )[0][0, [0, 2]].numpy()
    np.testing.assert_allclose(cov.sum(0), direct, atol=1e-12)
    assert downstream.shape == (7, 3)
    assert calls.count(("first", "attempted")) == 4
    assert calls.count(("first", "completed")) == 4
    assert all(k == "first" for k, _ in calls)
    jac = np.stack(
        [
            torch.autograd.grad(v, x, retain_graph=True)[0][0, [0, 2]].numpy()
            for v in logits
        ]
    )
    np.testing.assert_allclose(cov.sum(0), np.tensordot(w, jac, axes=1), atol=1e-12)


def test_bypass_fails_physical_action_gate():
    x, s, logits, masks = graph(True)
    cov, _ = D.route_covectors(
        logits, s, x, [0, 2], masks, np.array([1.0, 2.0]), lambda *_: None
    )
    direct = torch.autograd.grad(logits[0] + 2 * logits[1], x)[0][0, [0, 2]].numpy()
    with pytest.raises(ValueError, match="replay gate"):
        D.require_replay(cov.sum(0), direct)


def test_detached_weight_is_not_a_moving_observable():
    x, s, logits, masks = graph()
    weight = logits.square()
    cov, _ = D.route_covectors(logits, s, x, [0, 2], masks, weight, lambda *_: None)
    frozen = torch.autograd.grad(
        (weight.detach() * logits).sum(), x, retain_graph=True
    )[0][0, [0, 2]].numpy()
    moving = torch.autograd.grad((weight * logits).sum(), x)[0][0, [0, 2]].numpy()
    np.testing.assert_allclose(cov.sum(0), frozen, atol=1e-12)
    assert not np.allclose(cov.sum(0), moving)


def factors():
    rng = np.random.default_rng(4301)
    c = np.eye(3) - np.ones((3, 3)) / 3
    jp, e, j, d = (
        rng.normal(size=(3, 4, 2)),
        rng.normal(size=(4, 2)),
        rng.normal(size=(3, 3, 2, 2)),
        rng.normal(size=(2, 2, 2)),
    )
    h = j.sum(0)
    link = ((c @ h.reshape(3, -1) @ d.reshape(2, -1).T).T @ jp.reshape(3, -1)).reshape(
        2, 4, 2
    )
    return (
        c,
        jp,
        e,
        j,
        d,
        {
            "delta_h": d,
            "delta_p": e,
            "h_jacobian": h,
            "link": link,
            "decoder": link * 2,
            "roundoff": link * 0.01,
            "parent_total": link * 3.01,
        },
    )


def test_gauge_units_and_two_arms_share_covector():
    c, jp, e, j, d, old = factors()
    w, _ = D.dual_weights(c, jp, e)
    v = np.tensordot(w, j, axes=(0, 1))
    accounts = D.scalar_accounts(v, w, w, old)
    for a in range(2):
        expected = (
            e.ravel() @ jp.reshape(3, -1).T @ c @ j.sum(0).reshape(3, -1) @ d[a].ravel()
        )
        np.testing.assert_allclose(accounts["raw_link"][a], expected, atol=1e-12)
    wg, _ = D.dual_weights(c, jp + 7, e)
    np.testing.assert_allclose(w, wg, atol=1e-12)
    np.testing.assert_allclose(np.tensordot(w, j + 5, axes=(0, 1)), v, atol=1e-12)
    we, _ = D.dual_weights(c, jp, e * 9)
    np.testing.assert_allclose(we, w * 9, atol=1e-12)


def test_float32_weight_error_and_distinct_arithmetic():
    c, jp, e, j, _d, old = factors()
    w64, w32 = D.dual_weights(c, jp, e)
    v = np.tensordot(w32.astype(float), j, axes=(0, 1))
    a = D.scalar_accounts(v, w64, w32, old)
    np.testing.assert_allclose(a["raw_link"] - a["link"], a["weight_delta"], atol=1e-12)
    np.testing.assert_allclose(a["arithmetic_delta"], 0, atol=1e-12)
    changed = v.copy()
    changed[0] += 0.001
    b = D.scalar_accounts(changed, w64, w32, old)
    np.testing.assert_allclose(a["weight_delta"], b["weight_delta"])
    assert not np.allclose(b["arithmetic_delta"], 0)
    assert a["weight_delta"].any()


def test_cancellation_signed_cross_terms_and_raw_not_corrected():
    row = dict.fromkeys(D.FIELDS, 0.0)
    row.update(
        body=3.0,
        early_query=-2.0,
        raw_link=1.0,
        link=0.99,
        weight_delta=0.004,
        arithmetic_delta=0.006,
        weight_correction=-0.004,
        arithmetic_correction=-0.006,
        full=0.99,
    )
    summary = D.moments([row])
    assert summary["body__early_query_cross"] == -12
    assert summary["raw_link_mean"] == 1 and summary["link_mean"] == 0.99
    assert sum(k.endswith("_cross") for k in summary) == 21
    assert summary["full_sum_error"] < 1e-15


def test_pair_means_contrasts_and_binding():
    rows = [
        {
            "query": "q",
            "role": "source",
            "cell": 0,
            "arm": "u",
            "wording": w,
            **dict.fromkeys(D.FIELDS, v),
        }
        for w, v in ((0, 2.0), (2, 6.0))
    ]
    m, d = D.pair_rows(rows, [0, 2])
    assert m["full"] == 4 and d["full"] == 2
    with pytest.raises(ValueError):
        D.pair_rows(rows[::-1], [0, 2])


@pytest.mark.parametrize(
    "change",
    [{"maximum_first_calls": 385}, {"absolute_tolerance": 1e-7}, {"cut_layer": 18}],
)
def test_scope_cannot_be_relaxed(change):
    spec = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/grain_math_dual_route_action_v1.json"
        ).read_text()
    )
    D.assert_scope(spec)
    with pytest.raises(ValueError):
        D.assert_scope({**spec, **change})


def test_source_barrier_and_attempted_budget():
    status = MIX.new_status()
    D.require_stage(status, True, False)
    with pytest.raises(ValueError):
        D.require_stage(status, False, False)
    status.update(status="SOURCE_CHECKPOINTED", committed_nodes=48)
    D.require_stage(status, False, True)
    status["first_attempted"] = 384
    with pytest.raises(RuntimeError):
        MIX.reserve(status, "first", 384)


def test_alternate_scalar_pairing():
    from scripts.verify_grain_math_dual_route_action import pairing

    x = np.array([1e16, 1.0, -1e16])
    assert pairing(x, np.ones(3)) == 1.0
    with pytest.raises(ValueError):
        pairing([np.nan], [0])


def test_fixed_weights_reject_nonfinite_and_shape():
    for c, p, e in (
        (np.eye(2), np.ones((3, 2)), np.ones(2)),
        (np.eye(2), np.full((2, 2), np.nan), np.ones(2)),
    ):
        with pytest.raises(ValueError):
            D.dual_weights(c, p, e)
