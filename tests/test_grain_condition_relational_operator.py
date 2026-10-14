#!/usr/bin/env python3
"""Discriminating conditional-relation toy worlds, not personality validation."""

import numpy as np
import pytest

from grain_research.construction import math_condition_relational_operator as M
from grain_research.construction import math_relational_target_operator as P


def test_all_five_table_preserves_oriented_account_axes():
    x = np.arange(2 * 4 * 4 * 31).reshape(2, 4, 4, 31)
    actual = M.target_accounts(x)
    assert actual.shape == (2, 4, 4)
    for r, a, b in np.ndindex(2, 4, 4):
        assert actual[r, a, b] == sum(x[r, a, b, m] for m in P.TARGET_MODES)
    with pytest.raises(ValueError):
        M.target_accounts(np.zeros((4, 4, 5)))


def material(kind="mixed"):
    """All32 profiles have two authors with a target-dependent paired deviation."""
    g = np.repeat(np.arange(32), 2)
    a = np.tile([-1.0, 1.0], 32)
    b = a * (2 * (g % 2) - 1)
    p = np.stack((0.5 + a / 4, 0.5 + b / 4), 1)
    tau = np.array([[[-1.0, 0.0], [1.0, 0.0]], [[-1.0, 0.0], [1.0, 0.0]]])
    c = tau[:, 0] + p[..., None] * (tau[:, 1] - tau[:, 0])
    e = np.stack((np.c_[np.zeros(len(a)), a], np.c_[np.zeros(len(a)), b]), 1)
    if kind == "choice":
        e[:] = 0
    elif kind == "expression":
        p[:] = 0.5
        c[:] = 0
    x = c + e
    return M.compile_bank(x, p, g, tau), P.compile_bank(x, g), x, p, g


def tensors(bank):
    """Literal primal matrices for independent comparison of the dual code."""
    return np.array(
        [
            [bank["a"][:, a].T @ (w[:, None] * bank["b"][:, b]) for w in bank["w"]]
            for a, b in M.LEGS
        ]
    )


@pytest.mark.parametrize("kind,account", [("choice", 0), ("expression", 3)])
def test_pure_mechanisms_are_distinguishable(kind, account):
    bank, _, *_ = material(kind)
    t = tensors(bank)
    assert np.linalg.norm(t[account, 0]) > 0
    assert np.max(np.abs(np.delete(t, account, 0))) == 0


def test_source_and_signed_cross_terms_preserve_parent():
    b, parent, *_ = material()
    t = tensors(b)
    parent_t = np.array(
        [parent["pair_a"].T @ (w[:, None] * parent["pair_b"]) for w in parent["pair_w"]]
    )
    np.testing.assert_allclose(t.sum(0), parent_t, atol=1e-15)
    ds = M.contractions(b, b)
    np.testing.assert_allclose(ds, np.einsum("amij,bmij->abm", t, t), atol=1e-15)
    np.testing.assert_allclose(
        ds.sum((0, 1)), P.contraction(parent, parent, "pair", "pair", "full", np.eye(2))
    )
    other = {**b, "b": -b["b"]}
    assert M.contractions(b, other).sum((0, 1))[0] < 0


def test_binary_choice_is_rank_one_with_known_scalar():
    b, *_ = material()
    scalar = b["w"] @ np.prod(b["proportion_residuals"], axis=1)
    da, db = b["templates"][:, 1] - b["templates"][:, 0]
    np.testing.assert_allclose(tensors(b)[0], scalar[:, None, None] * np.outer(da, db))


def test_reference_changes_allocation_not_observation_or_total():
    b, parent, x, p, g = material()
    alternate = b["templates"].copy()
    alternate[:, 1, 1] += 4
    bb = M.compile_bank(x, p, g, alternate)
    assert np.linalg.norm(tensors(bb) - tensors(b)) > 0
    np.testing.assert_allclose(tensors(bb).sum(0), tensors(b).sum(0), atol=1e-15)
    for bank in (b, bb):
        obs = M.parts(x, p, bank["templates"])
        values = M.read(bank, obs[:, 0], obs[:, 1])
        np.testing.assert_allclose(
            values.sum((1, 2)),
            P.read(parent, "pair", "full", x[:, 0], x[:, 1], np.eye(2)),
            atol=1e-15,
        )


def test_moving_condition_and_expression_together_keeps_reference_total():
    b, parent, *_ = material()
    rp, ix = P.repaire(parent, np.random.default_rng(208))
    ds = M.contractions(b, b, ix, ix)
    np.testing.assert_allclose(
        ds.sum((0, 1)),
        P.contraction(rp, rp, "pair", "pair", "full", np.eye(2)),
        atol=1e-15,
    )


def test_label_free_templates_and_single_condition_coverage():
    v = np.arange(48, dtype=float).reshape(2, 2, 6, 2)
    c = np.tile([0, 0, 0, 1, 1, 1], (2, 2, 1))
    tau = M.templates(v, c)
    for view, z in np.ndindex(2, 2):
        np.testing.assert_allclose(tau[view, z], v[:, view][c[:, view] == z].mean(0))
    means = v.mean(2)
    x = M.parts(means, np.zeros((2, 2)), tau)
    np.testing.assert_allclose(x.sum(2), means)
    with pytest.raises(ValueError):
        M.templates(v, np.zeros_like(c))


def test_field_matches_edit_including_changed_condition_choice():
    b, _, x, p, _ = material()
    obs = M.parts(x[0], p[0], b["templates"])
    for view in (0, 1):
        new_x, new_p = x[0].copy(), p[0].copy()
        new_x[view] += [0.1, -0.2]
        new_p[view] += 0.1
        new = M.parts(new_x, new_p, b["templates"])
        delta = new[view] - obs[view]
        field = M.field(b, obs[1 - view], view)
        prediction = np.empty((4, 4, 31))
        for j, (c, d) in enumerate(M.LEGS):
            prediction[:, j] = field[:, j] @ delta[c if view == 0 else d]
        np.testing.assert_allclose(
            M.read(b, new[0], new[1]) - M.read(b, obs[0], obs[1]),
            prediction,
            atol=1e-15,
        )


def test_cold_refuses_missing_conditions_labels_and_repeated_events():
    b, *_ = material()
    record = {
        "frame_key": "toy",
        "windows": [
            [{"event_id": "a", "vector": [0.1, 0.4], "condition": 0}],
            [{"event_id": "b", "vector": [-0.2, 0.3], "condition": 1}],
        ],
    }
    assert np.asarray(M.cold(b, record)["source_by_observation_readings"]).shape == (
        4,
        4,
        31,
    )
    record["windows"][1][0]["event_id"] = "a"
    with pytest.raises(ValueError):
        M.cold(b, record)
    record["windows"][1][0]["event_id"] = "b"
    del record["windows"][1][0]["condition"]
    with pytest.raises(ValueError):
        M.cold(b, record)
    record["target"] = 0
    with pytest.raises(ValueError):
        M.cold(b, record)
