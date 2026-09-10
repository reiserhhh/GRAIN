#!/usr/bin/env python3
"""Finite identities for conditional baseline derivative transport."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.joint_readout import centered_logits
from grain_research.vector_mechanisms.query_baseline import (
    aggregate_jacobian,
    contract,
    predict_affine,
    source_affine,
)


def test_repeated_means_need_sum_not_mean():
    rng = np.random.default_rng(32)
    jac = rng.normal(size=(3, 13, 4))
    groups = [np.array([1, 2]), np.array([5, 6, 7, 8])]
    rest = np.array([0, 3, 4, 9])
    jm, jr = aggregate_jacobian(jac, groups, rest)
    dm = rng.normal(size=(2, 4))
    dr = rng.normal(size=(len(rest), 4))
    direction = np.zeros((13, 4))
    for positions, d in zip(groups, dm, strict=True):
        direction[positions] = d
    direction[rest] = dr
    np.testing.assert_allclose(
        contract(jac, direction), contract(jm, dm) + contract(jr, dr)
    )
    wrong = np.stack([jac[:, g].mean(1) for g in groups], 1)
    assert not np.allclose(
        contract(jac, direction), contract(wrong, dm) + contract(jr, dr)
    )


def test_affine_compilation_and_exact_linear_transfer():
    rng = np.random.default_rng(11)
    m = rng.normal(size=(4, 2, 3))
    r = rng.normal(size=(4, 4, 3))
    gm = rng.normal(size=(5, 2, 3))
    gr = rng.normal(size=(5, 4, 3))
    b = rng.normal(size=5)
    z = np.array(
        [b + contract(gm, a) + contract(gr, c) for a, c in zip(m, r, strict=True)]
    )
    model = source_affine(m, r, z, np.repeat(gm[None], 4, 0), np.repeat(gr[None], 4, 0))
    heldm = rng.normal(size=(8, 2, 3))
    heldr = rng.normal(size=(8, 4, 3))
    p = predict_affine(heldm, heldr, model)
    actual = centered_logits(
        np.array(
            [
                b + contract(gm, a) + contract(gr, c)
                for a, c in zip(heldm, heldr, strict=True)
            ]
        )
    )
    np.testing.assert_allclose(p["logits"], actual, atol=5e-15)
    np.testing.assert_allclose(p["per_source"].mean(1), p["logits"], atol=5e-15)


def test_local_derivative_not_global_guarantee():
    m = np.array([[[1.0]], [[2.0]]])
    r = np.zeros((2, 1, 1))
    z = np.array([[1.0, -1.0], [4.0, -4.0]])
    gm = np.array([[[[2.0]], [[-2.0]]], [[[4.0]], [[-4.0]]]])
    gr = np.zeros((2, 2, 1, 1))
    p = predict_affine(
        np.array([[[4.0]]]), np.zeros((1, 1, 1)), source_affine(m, r, z, gm, gr)
    )
    assert not np.allclose(p["logits"], [[16.0, -16.0]])


def test_invalid_jacobian_partition():
    with pytest.raises(ValueError):
        aggregate_jacobian(np.ones((2, 4, 3)), [np.array([0, 1])], np.array([1]))


def test_invalid_contract():
    with pytest.raises(ValueError):
        contract(np.ones((2, 3, 4)), np.ones((2, 4)))


def test_native_jacobian_on_known_frozen_linear_observer():
    from types import SimpleNamespace

    import torch

    from grain_research.vector_mechanisms.query_baseline import candidate_jacobian

    class Pool(torch.nn.Module):
        def forward(self, h, **kwargs):
            return h.mean(1, keepdim=True).expand_as(h)

    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([Pool()])
    inner.norm = torch.nn.Identity()
    model = torch.nn.Module()
    model.model = inner
    model.lm_head = torch.nn.Linear(4, 3, bias=False)
    model.requires_grad_(False)
    observer = SimpleNamespace(model=model, calls=0)
    callback = []
    _, jm, jr = candidate_jacobian(
        observer,
        torch.zeros(1, 7, 4),
        {},
        {"token_ids": [0, 1, 2]},
        [np.array([1, 2]), np.array([3, 4, 5])],
        np.array([0]),
        block=-1,
        on_backward=lambda: callback.append(1),
    )
    w = model.lm_head.weight.numpy()
    w = w - w.mean(0)
    np.testing.assert_allclose(jm[:, 0], w * 2 / 7, atol=1e-7)
    np.testing.assert_allclose(jm[:, 1], w * 3 / 7, atol=1e-7)
    np.testing.assert_allclose(jr[:, 0], w / 7, atol=1e-7)
    assert observer.calls == 1 and len(callback) == 3
