#!/usr/bin/env python3
"""Synthetic mechanism tests; no natural psychological validity is assumed."""

import numpy as np
import pytest

from grain_research.construction import math_relational_target_operator as M


def material(reverse=False):
    """Every profile has zero means, but E controls the cross-window relation."""
    g = np.repeat(np.arange(32), 2)
    a = np.tile([-1.0, 1.0], 32)
    sign = 2 * (g % 2) - 1
    b = a * sign * (-1 if reverse else 1)
    x = np.stack((np.c_[a, np.zeros(len(a))], np.c_[b, a]), 1)
    return M.compile_bank(x, g)


def primal(bank, component, space="full", basis=None):
    """Literal small-dimensional tensor, distinct from the production dual form."""
    if basis is None:
        basis = np.eye(2)[:, :1]
    a, b, w = M.sides(bank, component, space, basis)
    return np.array([a.T @ (row[:, None] * b) for row in w])


def test_relation_without_mean_and_profile_account():
    bank = material()
    assert np.max(np.abs(primal(bank, "profile"))) == 0
    assert primal(bank, "pair")[0, 0, 0] == pytest.approx(1)
    np.testing.assert_allclose(
        primal(bank, "joint"), primal(bank, "profile") + primal(bank, "pair")
    )


def test_negative_cross_family_is_retained():
    dot = M.contraction(material(), material(True), "pair", "pair", "full", np.eye(2))
    assert dot[0] < 0


def test_dual_primal_read_field_and_gram():
    b, q, x, y = (
        material(),
        np.eye(2)[:, :1],
        np.array([0.3, 0.8]),
        np.array([-0.4, 0.2]),
    )
    for component in M.COMPONENTS:
        for space in M.SPACES:
            t = primal(b, component, space, q)
            np.testing.assert_allclose(
                M.read(b, component, space, x, y, q),
                np.einsum("i,mij,j->m", x, t, y),
                atol=1e-15,
            )
            np.testing.assert_allclose(
                M.field(b, component, space, y, 0, q), t @ y, atol=1e-15
            )
            np.testing.assert_allclose(
                M.contraction(b, b, component, component, space, q),
                np.sum(t * t, axis=(1, 2)),
                atol=1e-15,
            )


def test_projected_channels_sum_and_coordinate_equivariance():
    b, q = material(), np.eye(2)[:, :1]
    t = primal(b, "joint")
    np.testing.assert_allclose(t, sum(primal(b, "joint", s, q) for s in M.SPACES[1:]))
    rot = np.array([[0.6, -0.8], [0.8, 0.6]])
    bb = {
        k: v @ rot if k.endswith(("_a", "_b")) or k == "centers" else v
        for k, v in b.items()
    }
    x, y = np.array([0.5, -0.1]), np.array([0.2, 0.8])
    np.testing.assert_allclose(
        M.read(b, "pair", "full", x, y, q),
        M.read(bb, "pair", "full", x @ rot, y @ rot, rot.T @ q),
        atol=1e-15,
    )


def test_pairing_exchange_keeps_profile_means():
    b = material()
    other, ix = M.repaire(b, np.random.default_rng(207))
    assert np.array_equal(b["codes"][ix], b["codes"])
    np.testing.assert_allclose(primal(other, "profile"), primal(b, "profile"))
    np.testing.assert_allclose(
        primal(other, "joint"), primal(other, "profile") + primal(other, "pair")
    )


def test_cold_refuses_repeated_events_targets_and_missing_windows():
    b = material()
    r = {
        "frame_key": "test",
        "windows": [
            [{"event_id": "a", "vector": [0.2, 0.1]}],
            [{"event_id": "b", "vector": [-0.2, 0.3]}],
        ],
    }
    assert np.asarray(M.cold(b, np.eye(2), r)["operator_readings"]).shape == (3, 31)
    r["windows"][1][0]["event_id"] = "a"
    with pytest.raises(ValueError):
        M.cold(b, np.eye(2), r)
    r["windows"][1] = []
    with pytest.raises(ValueError):
        M.cold(b, np.eye(2), r)
    r["target"] = 1
    with pytest.raises(ValueError):
        M.cold(b, np.eye(2), r)


def test_missing_profile_is_not_imputed():
    with pytest.raises(ValueError):
        M.compile_bank(np.zeros((2, 2, 2)), np.array([0, 1]))


def test_companion_field_agrees_with_finite_action_in_both_windows():
    b, q = material(), np.eye(2)[:, :1]
    x, y, delta = np.array([0.2, 0.4]), np.array([-0.3, 0.1]), np.array([0.1, -0.05])
    for component in M.COMPONENTS:
        for space in M.SPACES:
            old = M.read(b, component, space, x, y, q)
            for view in (0, 1):
                new = M.read(
                    b,
                    component,
                    space,
                    x + delta if view == 0 else x,
                    y + delta if view == 1 else y,
                    q,
                )
                companion = y if view == 0 else x
                np.testing.assert_allclose(
                    new - old,
                    M.field(b, component, space, companion, view, q) @ delta,
                    atol=1e-15,
                )
