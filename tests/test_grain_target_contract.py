#!/usr/bin/env python3
"""Mechanism controls for evidence-dependent target specification."""

import numpy as np
import pytest

from grain_research.construction import math_target_contract as M


def test_name_only_rival():
    y = np.zeros((3, 2, 3, 2))
    y[:, 0, 0] = 3
    y[:, 0, 1] = 1
    a = M.action_accounts(y)
    assert np.all(a["named_name"] == 2)
    assert not np.any(a["definition_by_name"])
    assert not np.any(a["interaction_corner"])


def test_definition_only_rival():
    y = np.zeros((3, 2, 3, 2))
    y[:, 0, :, 0] = 3
    y[:, 0, :, 1] = 1
    a = M.action_accounts(y)
    assert np.all(a["definition_by_name"] == 2)
    assert not np.any(a["named_name"])
    assert not np.any(a["interaction_corner"])


def test_interaction_not_silently_dropped():
    y = np.zeros((2, 2, 3, 2))
    y[:, 0, :2] = np.outer(M.SIGNS, M.SIGNS)
    a = M.action_accounts(y)
    assert np.all(a["interaction_corner"] == 4)
    assert not np.any(a["named_definition"])
    assert not np.any(a["named_name"])


def test_calibration_offsets_cancel_without_being_removed():
    y = np.random.default_rng(1).normal(size=(5, 2, 3, 2))
    offset = np.random.default_rng(2).normal(size=(5, 1, 3, 2))
    np.testing.assert_allclose(
        M.action_accounts(y)["delta"], M.action_accounts(y + offset)["delta"]
    )


def test_scalar_and_vector_reconstruction():
    for shape in ((4, 2, 2), (4, 2, 2, 13)):
        a = np.random.default_rng(4).normal(size=shape)
        co, rebuilt = M.factorial(a)
        np.testing.assert_allclose(a, rebuilt, atol=1e-14)
        np.testing.assert_allclose((a * a).mean((1, 2)), (co * co).sum(1))


def test_shared_logit_gauge_and_nonlinear_account():
    z = np.random.default_rng(5).normal(size=(2, 2, 3, 2, 10))
    shift = np.random.default_rng(6).normal(size=z.shape[:-1] + (1,))
    y, linear, curvature = M.responses(z)
    np.testing.assert_allclose(y, linear + curvature, atol=1e-14)
    np.testing.assert_allclose(y, M.responses(z + shift)[0], atol=1e-14)
    assert np.any(abs(curvature) > 0.01)


def test_same_linear_head_contraction():
    h = np.random.default_rng(7).normal(size=(4, 2, 3, 2, 9))
    w = np.random.default_rng(8).normal(size=9)
    vector = M.action_accounts(h)
    scalar = M.action_accounts(h @ w)
    for key in ("coefficients", "delta", "anonymous_definition", "interaction_corner"):
        np.testing.assert_allclose(vector[key] @ w, scalar[key], atol=1e-14)


def test_body_kept_and_definitions_named_separately():
    texts = [f"original addressed excerpt {i}" for i in range(4)]
    bodies = {
        M.render(texts, n, d).split("Question:")[0] for n in range(3) for d in range(2)
    }
    assert len(bodies) == 1
    crossed = M.render(texts, 0, 1)
    assert "Scale name: Agreeableness" in crossed
    assert "explores unfamiliar ideas" in crossed
    assert "voluntarily helps" not in crossed
    assert all(t in crossed for t in texts)
    assert "No excerpts" in M.render([], 2, 0)


def test_missing_or_invalid_scope_rejected():
    with pytest.raises(ValueError):
        M.action_accounts(np.zeros((5, 2, 2, 2)))
    with pytest.raises(ValueError):
        M.factorial(np.zeros((2, 2, 3)))
    with pytest.raises(ValueError):
        M.render(["a"], 0, 0)
    with pytest.raises(ValueError):
        M.contract(3, 0)
    with pytest.raises(ValueError):
        M.responses(np.zeros((4, 9)))
