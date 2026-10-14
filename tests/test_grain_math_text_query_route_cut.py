#!/usr/bin/env python3
"""Complete cut, bypass, physical contraction and source barrier witnesses."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grain_research.construction import math_mixed_query_path as MIX
from grain_research.construction import math_text_query_route_cut as R


def row():
    """Use noncontiguous query positions so remainder is not just endpoint."""
    return {
        "input_ids": list(range(7)),
        "question_start": 3,
        "early_query_positions": [3, 5],
        "spans": {"A_property": {"positions": [0, 2]}},
    }


def test_complete_position_masks():
    masks = R.route_masks(row())
    np.testing.assert_array_equal(masks.sum(0), 1)
    np.testing.assert_array_equal(np.flatnonzero(masks[2]), [4, 6])
    assert masks[0, 0] and masks[0, 2] and not masks[1, -1]


@pytest.mark.parametrize(
    "change",
    [
        {"early_query_positions": [3, 3]},
        {"early_query_positions": [3, 6]},
        {"early_query_positions": []},
        {"question_start": 0},
        {"spans": {"A_property": {"positions": [0, 3]}}},
    ],
)
def test_bad_addresses_fail(change):
    with pytest.raises(ValueError):
        R.route_masks({**row(), **change})


def graph(bypass=False):
    """Build a nonlinear full-state cut with residual and cross-position branches."""
    x = torch.arange(21, dtype=torch.float64).reshape(1, 7, 3).div(30).requires_grad_()
    s = x + torch.sin(x + x.flip(1)) + 0.2 * x.mean(1, keepdim=True)
    logits = torch.stack([s.square().sum(), (s[:, 0] * s[:, -1]).sum()])
    if bypass:
        logits = logits + x[:, 0].sum()
    return x, s, logits


def test_nonlinear_residual_chain_is_first_derivative():
    x, s, logits = graph()
    calls = []
    jac, downstream = R.route_jacobians(
        logits, s, x, [0, 2], R.route_masks(row()), lambda *a: calls.append(a)
    )
    direct = np.stack(
        [
            torch.autograd.grad(v, x, retain_graph=True)[0][0, [0, 2]].numpy()
            for v in logits
        ]
    )
    np.testing.assert_allclose(jac.sum(0), direct, atol=1e-12)
    assert downstream.shape == (2, 7, 3)
    assert calls.count(("first", "attempted")) == 8
    assert calls.count(("first", "completed")) == 8
    assert all(k == "first" for k, _ in calls)


def test_incomplete_graph_cut_detects_bypass():
    x, s, logits = graph(True)
    jac, _ = R.route_jacobians(
        logits, s, x, [0, 2], R.route_masks(row()), lambda *_: None
    )
    direct = np.stack(
        [
            torch.autograd.grad(v, x, retain_graph=True)[0][0, [0, 2]].numpy()
            for v in logits
        ]
    )
    with pytest.raises(ValueError, match="replay gate"):
        R.require_replay(jac.sum(0), direct)


def factors():
    rng = np.random.default_rng(4201)
    return (
        rng.normal(size=(3, 3, 2, 4)),
        rng.normal(size=(2, 2, 4)),
        rng.normal(size=(3, 3)),
        rng.normal(size=(3, 5, 4)),
    )


def test_ordered_links_reconstruct_parent_and_alternate():
    from scripts.verify_grain_math_text_query_route_cut import ordered_routes

    j, d, c, p = factors()
    out = R.link_routes(j, d, c, p)
    direct = (
        p.reshape(3, -1).T @ c @ j.sum(0).reshape(3, -1) @ d.reshape(2, -1).T
    ).T.reshape(2, 5, 4)
    np.testing.assert_allclose(out.sum(0), direct, atol=1e-12)
    np.testing.assert_allclose(out, ordered_routes(j, d, c, p), atol=1e-12)


def test_gauge_and_physical_units():
    j, d, _, p = factors()
    c = np.eye(3) - np.ones((3, 3)) / 3
    expected = R.link_routes(j, d, c, p)
    np.testing.assert_allclose(R.link_routes(j + 7, d, c, p + 5), expected, atol=1e-12)
    np.testing.assert_allclose(R.link_routes(j, d * 9, c, p), expected * 9, atol=1e-12)
    np.testing.assert_allclose(
        R.link_routes(j / 4, d * 4, c * 2, p), expected * 2, atol=1e-12
    )


def test_route_cancellation_and_all_signed_crosses():
    data = dict.fromkeys(R.FIELDS, 0.0)
    data.update(body=3.0, early_query=-2.0, link=1.0, full=1.0)
    summary = R.moments([data])
    assert summary["body_ms"] == 9 and summary["early_query_ms"] == 4
    assert summary["body__early_query_cross"] == -12
    assert summary["full_ms"] == 1 and summary["full_sum_error"] == 0
    assert sum(k.endswith("_cross") for k in summary) == 15


def test_pair_means_and_half_contrasts():
    rows = [
        {
            "query": "q",
            "role": "source",
            "cell": 0,
            "arm": "a",
            "wording": w,
            **dict.fromkeys(R.FIELDS, v),
        }
        for w, v in ((0, 2.0), (2, 6.0))
    ]
    mean, contrast = R.pair_rows(rows, [0, 2])
    assert mean["full"] == 4 and contrast["full"] == 2
    with pytest.raises(ValueError):
        R.pair_rows(rows[::-1], [0, 2])


def test_scope_source_barrier_and_journal_caps():
    spec = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/grain_math_text_query_route_cut_v1.json"
        ).read_text()
    )
    R.assert_scope(spec)
    for key, value in (
        ("cut_layer", 18),
        ("maximum_first_calls", 3841),
        ("absolute_tolerance", 1e-7),
    ):
        with pytest.raises(ValueError):
            R.assert_scope({**spec, key: value})
    s = MIX.new_status()
    R.require_stage(s, True, False)
    with pytest.raises(ValueError):
        R.require_stage(s, False, False)
    s.update(status="SOURCE_CHECKPOINTED", committed_nodes=48)
    R.require_stage(s, False, True)
    with pytest.raises(ValueError):
        R.require_stage(s, True, True)
    s["first_attempted"] = 3840
    with pytest.raises(RuntimeError):
        MIX.reserve(s, "first", 3840)


def test_physical_roundoff_is_recorded_not_hidden():
    routes = np.zeros((3, 2, 2, 2))
    old = {
        "delta_p": np.ones((2, 2)),
        "link": np.ones((2, 2, 2)),
        "decoder": np.full((2, 2, 2), 2.0),
        "roundoff": np.full((2, 2, 2), 0.25),
        "parent_total": np.full((2, 2, 2), 3.25),
    }
    path = {
        "path": 0,
        "query": "q",
        "role": "source",
        "H": 0,
        "B": 0,
        "Q": 0,
        "wording": 0,
    }
    rows, residual = R.action_rows(path, routes, old, list(R.GROUPS), ["u", "m"])
    np.testing.assert_array_equal(residual, 1)
    assert rows[0]["cut_roundoff"] == 4 and rows[0]["full"] == 13
    assert rows[0]["full_sum_error"] == 0


def test_nonfinite_or_changed_shape_replay_fails():
    for actual, parent in (([np.nan], [0]), ([1, 2], [1]), ([1], [0])):
        with pytest.raises(ValueError):
            R.require_replay(actual, parent)


def test_cpu_cap_saves_failure(tmp_path):
    from scripts.run_grain_math_text_query_route_cut import cpu_charge

    (tmp_path / "run_status.json").write_text(
        json.dumps({"status": "PREPARED", "analysis_seconds": 180.0})
    )
    with pytest.raises(TimeoutError):
        cpu_charge(tmp_path, started=0.0)
    assert (
        json.loads((tmp_path / "run_status.json").read_text())["status"]
        == "STOPPED_FAILURE_OR_BUDGET"
    )
