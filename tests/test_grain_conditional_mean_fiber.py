#!/usr/bin/env python3
"""Synthetic proof checks for conditional-law aliases and joint-moment recovery."""

import numpy as np

from grain_research.construction import math_conditional_mean_fiber as M


def test_positive_alias_preserves_marginals_and_changes_joint_target():
    g = np.array([[b, c] for b in (-1, 1) for c in (-1, 1)], float)
    y = g[:, 0] * g[:, 1]
    d = M.conditional_contrasts(1)
    null = M.conditional_null_space(g, d)
    assert null["nullity"] == 1
    w = M.positive_witness(y, d, null["basis"])
    assert w["status"] == "POSITIVE_CONDITIONAL_MEAN_FIBER"
    assert np.min(w["plus"]) > 0 and np.min(w["minus"]) > 0
    np.testing.assert_allclose(w["plus"].sum(), 1)
    np.testing.assert_allclose(w["minus"].sum(), 1)
    np.testing.assert_allclose((w["plus"] - w["minus"]) @ g, 0, atol=1e-14)
    assert w["difference"] > 0
    full = M.conditional_null_space(M.summary_columns(g, True), d)
    assert full["nullity"] == 0
    assert M.positive_witness(y, d, full["basis"])["plus"] is None


def test_every_stratum_margin_is_zero_and_rotation_preserves_null_space():
    rng = np.random.default_rng(1371)
    d = M.conditional_contrasts(7)
    g = rng.normal(size=(28, 4))
    alpha = rng.normal(size=7)
    a = (d @ alpha).reshape(7, 2, 2)
    np.testing.assert_allclose(a.sum(axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(a.sum(axis=2), 0, atol=1e-14)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    left, right = M.conditional_null_space(g, d), M.conditional_null_space(g @ q, d)
    np.testing.assert_allclose(
        left["basis"] @ left["basis"].T, right["basis"] @ right["basis"].T, atol=1e-13
    )


def test_mean_identified_target_is_not_forced_to_have_a_witness():
    rng = np.random.default_rng(1372)
    d = M.conditional_contrasts(5)
    g = rng.normal(size=(20, 3))
    null = M.conditional_null_space(g, d)
    y = g @ np.array([2.0, -3.0, 1.0]) + 4
    assert M.positive_witness(y, d, null["basis"])["plus"] is None
