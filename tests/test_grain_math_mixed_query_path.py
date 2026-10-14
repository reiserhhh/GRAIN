#!/usr/bin/env python3
"""Analytic HM35 mixed paths, signed support and finite call accounting."""

import numpy as np
import pytest

from grain_research.construction import math_mixed_query_path as M


def test_smooth_mixed_derivative_integral():
    """A true autograd mixed derivative integrates a polynomial H response."""
    import torch

    values = []
    for t in M.NODES:
        leaf = torch.tensor([2.0, t], dtype=torch.float64, requires_grad=True)
        score = leaf[0] * (3 * leaf[1] ** 3 + 2 * leaf[1])
        alpha = torch.autograd.grad(score, leaf, create_graph=True)[0][0]
        mixed = torch.autograd.grad(alpha, leaf)[0][1].item()
        values.append(mixed)
    q = M.quadrature(values)
    assert q["simpson"] == pytest.approx(2.5)
    assert q["trapezoid"] != pytest.approx(2.5)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_fixed_half_integral_rules(degree):
    v = np.array(M.NODES) ** degree
    assert M.quadrature(v)["simpson"] == pytest.approx(0.5 / (degree + 1))


def test_same_full_path_signed_cancellation():
    v = np.array([[[1.0, -0.9]]])
    delta, support = np.ones((1, 2)), np.array([[1.0, 0]])
    action = M.mixed_actions(v, delta, support, delta - support)
    assert action[0] == pytest.approx([0.1, 1, -0.9])
    integrated = M.quadrature(np.stack([action] * 3))["simpson"]
    assert integrated[0] == pytest.approx([0.05, 0.5, -0.45])
    assert 2 * integrated[0, 1] * integrated[0, 2] < 0


def test_frozen_projector_and_token_identity():
    d = np.array([[1.0], [0]])
    model = {
        "D": d,
        "gram_inverse": np.ones((1, 1)),
        "p_shape": [1, 2],
        "query_token_ids": [3],
    }
    s, u = M.project(model, np.array([[2.0, 4]]), [3])
    assert s.tolist() == [[2, 0]]
    assert u.tolist() == [[0, 4]]
    with pytest.raises(ValueError, match="coordinates"):
        M.project(model, np.array([[2.0, 4]]), [4])


def test_float32_endpoints_preserved():
    a = np.array([[1.2, -7.01]], np.float32)
    b = np.array([[0.3, 8.2]], np.float32)
    assert np.array_equal(M.interpolate(a, b, 0), b)
    assert np.array_equal(M.interpolate(a, b, 1), a)
    assert M.interpolate(a, b, 0.5).dtype == np.float32
    with pytest.raises(ValueError):
        M.interpolate(a, b, 0.25)


def test_uncertain_second_call_reserves_budget():
    status = M.reserve(M.new_status(), "second", 1)
    assert status["second_attempted"] == 1
    assert status["second_completed"] == 0
    with pytest.raises(RuntimeError, match="budget"):
        M.reserve(status, "second", 1)


def test_design_source_then_application_and_bad_ids():
    rows = []
    for p, role in ((0, "source"), (3, "application")):
        for w in (0, 2):
            for h in (0, 1):
                for b in (0, 1):
                    for early in (0, 1):
                        for q in (0, 1):
                            rows.append(
                                {
                                    "pair": p,
                                    "wording": w,
                                    "query": "x",
                                    "role": role,
                                    "donor_side": h,
                                    "receiver_side": b,
                                    "early_query_side": early,
                                    "query_side": q,
                                    "input_ids": [2, 3],
                                    "early_query_token_ids": [2],
                                }
                            )
    result = M.design(rows, ["x"])
    assert len(result) == 32
    assert {r["role"] for r in result[:16]} == {"source"}
    assert {r["role"] for r in result[16:]} == {"application"}
    rows[1]["early_query_token_ids"] = [99]
    with pytest.raises(ValueError):
        M.design(rows, ["x"])


@pytest.mark.parametrize("bad", [np.zeros(2), np.full(3, np.nan)])
def test_reject_incomplete_quadrature(bad):
    with pytest.raises(ValueError):
        M.quadrature(bad)
