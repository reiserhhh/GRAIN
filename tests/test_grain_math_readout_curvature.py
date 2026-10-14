#!/usr/bin/env python3
"""Analytic and actual double-autograd tests before the bounded HM39 capture."""

import numpy as np
import pytest

from grain_research.construction import math_mixed_query_path as M
from grain_research.construction import math_readout_curvature as R


def toy(nonlinear=False, linear_score=False, detached=True):
    """A small CPU decoder with independently differentiated scalar composition."""
    import torch

    x = torch.tensor([0.4, -0.7], dtype=torch.float64, requires_grad=True)
    l = torch.stack(
        (x[0] + 2 * x[1], -x[0] + x[1] + (x[0] * x[1] if nonlinear else 0), x[0] - x[1])
    )
    a = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    p = torch.softmax(l, 0)
    f = l @ a if linear_score else p @ a
    g = torch.autograd.grad(f, l, create_graph=True, retain_graph=True)[0]
    total = torch.autograd.grad(
        torch.autograd.grad(f, x, create_graph=True, retain_graph=True)[0][0],
        x,
        retain_graph=True,
    )[0][1]
    jac = torch.stack([torch.autograd.grad(lj, x, retain_graph=True)[0] for lj in l])
    weighted = l @ (g.detach() if detached else g)
    first = torch.autograd.grad(weighted, x, create_graph=True, retain_graph=True)[0][0]
    decoder = (
        torch.autograd.grad(first, x)[0][1]
        if first.requires_grad
        else torch.zeros((), dtype=x.dtype)
    )
    h = (
        np.zeros((3, 3))
        if linear_score
        else R.link_derivatives(l.detach().numpy(), a.numpy())[3]
    )
    link = jac[:, 1].numpy() @ h @ jac[:, 0].numpy()
    return float(total.detach()), float(decoder.detach()), float(link)


def test_affine_decoder_zero_internal_curvature():
    total, decoder, link = toy()
    assert decoder == pytest.approx(0, abs=1e-14)
    assert total == pytest.approx(link, abs=1e-14)


def test_nonlinear_decoder_exact_chain():
    total, decoder, link = toy(nonlinear=True)
    assert abs(decoder) > 0.01
    assert total == pytest.approx(decoder + link, abs=1e-14)


def test_linear_score_has_only_decoder_curvature():
    total, decoder, link = toy(nonlinear=True, linear_score=True)
    assert link == 0
    assert total == pytest.approx(decoder, abs=1e-14)


def test_not_detaching_weights_is_not_decoder_term():
    _, decoder, _ = toy(nonlinear=True)
    _, wrong, _ = toy(nonlinear=True, detached=False)
    assert abs(wrong - decoder) > 0.01


def test_indefinite_and_common_shift_quotient():
    _, _, _, h = R.link_derivatives([0.0, 0.0, 0.0], [0.0, 0.5, 1.0])
    eigen = np.linalg.eigvalsh(h)
    assert eigen.min() < -0.01 and eigen.max() > 0.01
    assert np.max(abs(h @ np.ones(3))) < 1e-14
    assert np.max(abs(h - h.T)) == 0


def test_constant_values_and_shift_invariance():
    s, p, g, h = R.link_derivatives([-1.0, 0.4, 2.0], [0.4] * 3)
    assert s == pytest.approx(0.4)
    assert max(np.max(abs(g)), np.max(abs(h))) < 1e-14
    for a, b in zip(
        R.link_derivatives([-1.0, 0.4, 2.0], [0.0, 0.5, 1.0]),
        R.link_derivatives([999.0, 1000.4, 1002.0], [0.0, 0.5, 1.0]),
        strict=True,
    ):
        assert np.asarray(a) == pytest.approx(b, abs=1e-12)
    assert p.sum() == pytest.approx(1)


def test_contraction_and_physical_units():
    jp = np.arange(12.0).reshape(3, 2, 2)
    dh, h = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]), np.eye(3)
    v = R.link_covectors(h, dh, jp)
    assert v[0] == pytest.approx(sum(dh[0, j] * jp[j] for j in range(3)))
    u = np.eye(4)[:, :2]
    a = R.project_components(v, u, np.ones((2, 2)), np.array([2.0, 3.0]))
    b = R.project_components(v, u, np.ones((2, 2)), np.array([4.0, 6.0]))
    assert a["supported"] == pytest.approx(b["supported"])
    assert a["theta"] == pytest.approx(2 * b["theta"])


def test_opposing_components_preserve_cancellation():
    row = {"link": 1.0, "decoder": -0.9, "roundoff": 0.0, "total": 0.1}
    m = R.metrics([row])
    assert m["link_ms_over_total_ms"] == pytest.approx(100)
    assert m["link_decoder_cross_over_total_ms"] == pytest.approx(-180)
    assert m["sum_identity_error"] < 1e-14


def test_token_layout_and_call_boundaries():
    row = {
        "spans": {"A_property": {"positions": [0]}},
        "early_query_positions": [1],
        "early_query_token_ids": [2],
        "input_ids": [1, 2, 3],
    }
    R.validate_layout(row, (1, 3), (1, 3))
    with pytest.raises(ValueError):
        R.validate_layout(row, (2, 3), (1, 3))
    status = M.reserve(M.new_status(), "first", 1)
    assert status["first_attempted"] == 1 and status["first_completed"] == 0
    with pytest.raises(RuntimeError):
        M.reserve(status, "first", 1)


def test_invalid_arrays_fail_closed():
    with pytest.raises(ValueError):
        R.link_derivatives([float("nan")], [0])
    with pytest.raises(ValueError):
        R.link_covectors(np.ones(3), np.ones((2, 3)), np.ones((3, 2, 2)))
    with pytest.raises(ValueError):
        R.project_components(np.zeros((2, 1, 2)), np.eye(2), np.ones((1, 2)), [0, 1])
