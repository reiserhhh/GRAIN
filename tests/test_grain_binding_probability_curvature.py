#!/usr/bin/env python3
"""A mixed probability derivative need not imply a mixed logit interaction."""

import numpy as np
import torch


def test_additive_margin_produces_rank_one_mixed_probability_curvature():
    """A confidence-only cross term survives without any x-by-y margin term."""
    a = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    b = torch.tensor([-0.7, 0.4, 1.3], dtype=torch.float64)
    x = torch.zeros(3, requires_grad=True, dtype=torch.float64)
    y = torch.zeros(3, requires_grad=True, dtype=torch.float64)
    margin = 6.6 + a @ x + b @ y
    p = torch.sigmoid(margin)
    gx = torch.autograd.grad(p, x, create_graph=True)[0]
    hp = torch.stack([torch.autograd.grad(v, y, retain_graph=True)[0] for v in gx])
    expected = p * (1 - p) * (1 - 2 * p) * torch.outer(a, b)
    np.testing.assert_allclose(hp.detach(), expected.detach(), atol=1e-15)
    assert np.linalg.matrix_rank(hp.detach().numpy()) == 1
    assert float(torch.linalg.norm(hp)) > 0
    # The raw margin is additive, so its mixed Hessian is exactly zero.
    hm = torch.autograd.functional.hessian(
        lambda xx, yy: 6.6 + a @ xx + b @ yy, (x, y)
    )[0][1]
    assert torch.count_nonzero(hm) == 0


def test_individually_balanced_margin_removes_probability_link_cross_term():
    """At p=0.5, logistic-link curvature alone cannot create a mixed derivative."""
    a = torch.tensor([1.0, -2.0], dtype=torch.float64)
    b = torch.tensor([0.4, 1.3], dtype=torch.float64)
    x = torch.zeros(2, requires_grad=True, dtype=torch.float64)
    y = torch.zeros(2, requires_grad=True, dtype=torch.float64)
    h = torch.autograd.functional.hessian(
        lambda xx, yy: torch.sigmoid(a @ xx + b @ yy), (x, y)
    )[0][1]
    assert torch.count_nonzero(h) == 0
