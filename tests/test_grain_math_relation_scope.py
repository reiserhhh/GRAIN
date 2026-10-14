#!/usr/bin/env python3
"""Finite relation laws can fail scalar action without changing any marginal."""

import numpy as np
import pytest

from grain_research.construction import math_relation_scope as M
from scripts.run_grain_relation_scope_hm91 import sentence


def test_complete_basis_and_reconstruction():
    points, basis, names = M.design_basis()
    assert "xg" in names and "yh" in names
    np.testing.assert_allclose(basis @ basis.T, 16 * np.eye(16))
    values = points[:, 0] * points[:, 2] + 2 * points[:, 1]
    coef = M.coefficients(values)
    assert coef[names.index("xg")] == 1
    assert coef[names.index("y")] == 2
    np.testing.assert_allclose(basis.T @ coef, values)


def test_scalar_and_non_scalar_actions():
    u = np.array([1.0, 2, 3])
    same = M.scalar_account(u, -2 * u)
    assert same["multiplier"] == -2
    assert same["non_scalar_energy_fraction"] == 0
    changed = M.scalar_account(u, np.array([2.0, -1, 0]))
    assert changed["non_scalar_energy_fraction"] == 1
    assert changed["identity_error"] < 1e-12
    assert not M.scalar_account(np.zeros(3), u)["observed"]


def test_seven_term_account_and_cancellation():
    old = (np.array([0.3, 0.7]), np.array([0.2, 0.9]), np.array([-2.0, 1]))
    new = (np.array([0.6, 0.4]), np.array([0.8, 0.1]), np.array([1.0, -3]))
    account = M.product_account(old, new)
    assert account["identity_error"] < 1e-12
    assert account["triple"] != 0


def test_relation_joint_is_not_any_marginal_signal():
    plus, minus = M.relation_laws([1, -1, 0, 0, 0, 0])
    np.testing.assert_allclose(plus.sum(0), minus.sum(0))
    np.testing.assert_allclose(plus.sum(1), minus.sum(1))
    x = M.design_basis()[0]
    z = (x[:, 0] * x[:, 2] + x[:, 1] * x[:, 3]) / 2
    surface = np.array([1, -1, 0, 0, 0, 0])[:, None] * z[None]
    assert np.sum((plus - minus) * surface) > 0
    for blind in (
        np.broadcast_to(z, surface.shape),
        np.broadcast_to(np.arange(6)[:, None], surface.shape),
    ):
        assert abs(np.sum((plus - minus) * blind)) < 1e-12
    with pytest.raises(ValueError, match="centered"):
        M.relation_laws([1, 0])


def test_scope_changes_order_not_word_multiset():
    for frame in ("present_report", "past_claim"):
        inner = sentence("K1", 1, "join a discussion", frame, "negated_event")
        outer = sentence("K1", 1, "join a discussion", frame, "negated_report")
        assert inner != outer
        assert sorted(inner.split()) == sorted(outer.split())
