#!/usr/bin/env python3
"""Bilateral chain, causal query, indefinite ordered action and scope witnesses."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grain_research.construction import math_bilateral_route_interaction as B
from grain_research.construction import math_mixed_query_path as MIX


def graph(bypass=False):
    """Causal nonlinear cumulative mixing with a residual path at every position."""
    x = torch.arange(21, dtype=torch.float64).reshape(1, 7, 3).div(30).requires_grad_()
    s = x + torch.tanh(x.cumsum(1))
    logits = torch.stack([s[:, -1].square().sum(), (s[:, 0] * s[:, -1]).sum()])
    if bypass:
        logits = logits + x[:, 0].sum()
    masks = np.zeros((3, 7), bool)
    masks[0, :3], masks[1, [3, 5]] = True, True
    masks[2] = ~masks[:2].any(0)
    return x, s, logits, masks


def test_both_patch_legs_from_one_vjp_and_causal_body_zero():
    x, s, l, m = graph()
    calls = []
    h, p, down = B.bilateral_jacobians(
        l, s, x, [0, 2], [3, 5], m, lambda *a: calls.append(a)
    )
    direct = [torch.autograd.grad(v, x, retain_graph=True)[0][0].numpy() for v in l]
    np.testing.assert_allclose(h.sum(0), np.array(direct)[:, [0, 2]], atol=1e-12)
    np.testing.assert_allclose(p.sum(0), np.array(direct)[:, [3, 5]], atol=1e-12)
    assert np.count_nonzero(p[0]) == 0
    assert down.shape == (2, 7, 3)
    assert (
        calls.count(("first", "attempted")) == calls.count(("first", "completed")) == 8
    )


def test_graph_bypass_is_not_invented_zero_route():
    x, s, l, m = graph(True)
    h, _, _ = B.bilateral_jacobians(l, s, x, [0, 2], [3, 5], m, lambda *_: None)
    direct = np.array(
        [torch.autograd.grad(v, x, retain_graph=True)[0][0, [0, 2]].numpy() for v in l]
    )
    with pytest.raises(ValueError, match="replay gate"):
        B.require_replay(h.sum(0), direct)


def factors():
    rng = np.random.default_rng(4401)
    return (
        rng.normal(size=(3, 3, 2, 4)),
        rng.normal(size=(3, 3, 5, 4)),
        rng.normal(size=(2, 2, 4)),
        rng.normal(size=(5, 4)),
        rng.normal(size=(3, 3)),
    )


def test_ordered_noncommuting_bilinear_action_and_alternate():
    from scripts.verify_grain_math_bilateral_route_interaction import ordered_joint

    h, p, d, e, c = factors()
    a = B.joint_actions(h, p, d, e, c)
    expected = np.array(
        [
            [
                [
                    d[i].ravel()
                    @ h[g].reshape(3, -1).T
                    @ c
                    @ p[k].reshape(3, -1)
                    @ e.ravel()
                    for k in range(3)
                ]
                for g in range(3)
            ]
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(a, expected, atol=1e-10)
    np.testing.assert_allclose(a, ordered_joint(h, p, d, e, c), atol=1e-10)
    assert not np.allclose(a, B.joint_actions(h, p, d, e, c.T))


def test_gauge_and_physical_units():
    h, p, d, e, _ = factors()
    c = np.eye(3) - np.ones((3, 3)) / 3
    a = B.joint_actions(h, p, d, e, c)
    np.testing.assert_allclose(B.joint_actions(h + 5, p - 7, d, e, c), a, atol=1e-10)
    np.testing.assert_allclose(B.joint_actions(h, p, d * 9, e, c), a * 9, atol=1e-10)
    np.testing.assert_allclose(B.joint_actions(h, p, d, e * 3, c), a * 3, atol=1e-10)


def test_indefinite_curvature_produces_signed_cancellation():
    h, p = np.zeros((3, 2, 1, 1)), np.zeros((3, 2, 1, 1))
    h[0, :, 0, 0], p[1, :, 0, 0], p[2, :, 0, 0] = [1, 1], [2, 0], [0, 3]
    a = B.joint_actions(h, p, np.ones((2, 1, 1)), np.ones((1, 1)), np.diag([1, -1]))
    assert a[0, 0, 1] == 2 and a[0, 0, 2] == -3 and a[0].sum() == -1


def test_marginals_do_not_identify_joint_even_with_causal_zero_column():
    a = np.array([[0.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 6.0, 7.0]])
    b = a + np.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(a.sum(0), b.sum(0))
    np.testing.assert_array_equal(a.sum(1), b.sum(1))
    assert not np.array_equal(a, b)


def test_raw_and_marginal_bookkeeping_all_crosses():
    h, p, d, e, c = factors()
    a = B.joint_actions(h, p, d, e, c)
    raw = a.sum((1, 2))
    # Synthetic old covectors use the same scalar target with an explicit discrepancy.
    link = (raw + 0.01)[:, None, None] * np.ones((2, 5, 4)) / (e.sum())
    old = {
        "delta_p": e,
        "link": link,
        "decoder": link * 2,
        "roundoff": link * 0.01,
        "parent_total": link * 3.01,
    }
    acc = B.scalar_accounts(a, old)
    path = {
        "path": 0,
        "query": "q",
        "role": "source",
        "wording": 0,
        "H": 0,
        "B": 0,
        "Q": 0,
    }
    rows = B.action_rows(path, acc, ["u", "m"])
    np.testing.assert_allclose(acc["cut_roundoff"], 0.01, atol=1e-12)
    assert rows[0]["raw_link"] != rows[0]["link"]
    result = B.moments(rows)
    assert result["full_sum_error"] < 1e-10
    assert result["H_margin_sum_error"] < 1e-10
    assert result["P_margin_sum_error"] < 1e-10
    assert sum(k.endswith("_cross") and "___" in k for k in result) == 66
    assert all(
        a + "___" + b + "_cross" in result
        for i, a in enumerate(B.CELLS)
        for b in B.CELLS[i + 1 :]
    )


def test_wording_mean_and_contrast_binding():
    rr = [
        {
            "query": "q",
            "role": "source",
            "cell": 0,
            "arm": "u",
            "wording": w,
            **dict.fromkeys(B.FIELDS, v),
        }
        for w, v in ((0, 2.0), (2, 6.0))
    ]
    m, d = B.pair_rows(rr, [0, 2])
    assert m["full"] == 4 and d["full"] == 2
    with pytest.raises(ValueError):
        B.pair_rows(rr[::-1], [0, 2])


@pytest.mark.parametrize(
    "change",
    [{"maximum_first_calls": 961}, {"absolute_tolerance": 1e-7}, {"source_paths": 48}],
)
def test_scope_cannot_be_relaxed(change):
    s = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/grain_math_bilateral_route_interaction_v1.json"
        ).read_text()
    )
    B.assert_scope(s)
    with pytest.raises(ValueError):
        B.assert_scope({**s, **change})


def test_actual_attempted_budget_and_source_barrier():
    s = MIX.new_status()
    B.require_stage(s, True, False)
    with pytest.raises(ValueError):
        B.require_stage(s, False, False)
    s.update(status="SOURCE_CHECKPOINTED", committed_nodes=12)
    B.require_stage(s, False, True)
    s["first_attempted"] = 960
    with pytest.raises(RuntimeError):
        MIX.reserve(s, "first", 960)


def test_near_zero_total_retains_actions_without_percentages():
    row = dict.fromkeys(B.FIELDS, 0.0)
    row.update({B.CELLS[1]: 1.0, B.CELLS[2]: -1.0})
    out = B.moments([row])
    assert out["raw_link_rms"] == 0 and out[B.CELLS[1] + "_ms"] == 1
    assert out[B.CELLS[1] + "___" + B.CELLS[2] + "_cross"] == -2
    assert all(np.isfinite(v) for v in out.values())


def test_nonfinite_and_mismatched_factor_shapes_fail():
    h, p, d, e, c = factors()
    with pytest.raises(ValueError):
        B.joint_actions(h, p, d[:1], e, c)
    c[0, 0] = np.nan
    with pytest.raises(ValueError):
        B.joint_actions(h, p, d, e, c)
