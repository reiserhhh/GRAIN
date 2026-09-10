#!/usr/bin/env python3
"""Conditional scalar compatibility, source reference and matched-view objectives."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import context_modulated_potential as c
from grain_research.vector_mechanisms import hermite_token_potential as h
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    source_descriptor,
)


def toy():
    """Create four source authors and a frozen small Hermite descriptor."""
    r = np.random.default_rng(640)
    bodies = [r.normal(size=(t, 6)) for t in (12, 16, 20, 24)]
    gradients = [r.normal(size=x.shape) / len(x) for x in bodies]
    lasts = r.normal(size=(4, 6))
    m = source_descriptor(bodies, gradients, list(range(4)), rank=2, landmarks=2)
    width = 1 + 4 + 8 * 3
    m |= {
        "coef": r.normal(size=width),
        "score_scale": np.array(1.3),
        "density_scale": np.array(0.8),
        "alpha": np.array(0.01),
        "origin": np.array(0.2),
    }
    full = m | c.context_reference(lasts, m) | {"arm": np.array(c.ARMS[1])}
    full["coef"] = r.normal(size=width + 24)
    return r, bodies, lasts, gradients, m, full


def test_nesting_reference_zero_and_frozen_body():
    _, bs, ls, _, base, full = toy()
    nested = dict(full, coef=np.r_[base["coef"], np.zeros(24)])
    for a, b in zip(c.predict(bs[0], ls[0], nested), h.predict(bs[0], ls[0], base)):
        np.testing.assert_allclose(a, b, atol=1e-12)
    atzero = c.predict(bs[0], full["context_mean"], full)
    prefix = dict(base, coef=full["coef"][: len(base["coef"])])
    for a, b in zip(atzero, h.predict(bs[0], full["context_mean"], prefix)):
        np.testing.assert_allclose(a, b, atol=1e-12)
    for key in ("basis", "center", "scale", "landmarks", "bandwidth"):
        np.testing.assert_array_equal(full[key], base[key])


def test_body_last_and_mixed_derivatives():
    r, bs, ls, _, _, m = toy()
    b, last, eps = bs[0], ls[0], 1e-5
    d, dl = r.normal(size=b.shape), r.normal(size=last.shape)
    body_grad = lambda x, l: hidden_gradient(c.predict(x, l, m)[1], m)
    exact = np.sum(body_grad(b, last) * d) + c.last_gradient(b, last, m) @ dl
    fd = (
        c.scalar(b + eps * d, last + eps * dl, m)
        - c.scalar(b - eps * d, last - eps * dl, m)
    ) / (2 * eps)
    assert fd == pytest.approx(exact, abs=1e-8)
    mixed_body = np.sum(
        (body_grad(b, last + eps * dl) - body_grad(b, last - eps * dl)) * d
    ) / (2 * eps)
    mixed_last = (
        (c.last_gradient(b + eps * d, last, m) - c.last_gradient(b - eps * d, last, m))
        @ dl
        / (2 * eps)
    )
    assert mixed_body == pytest.approx(mixed_last, abs=1e-8)
    assert abs(mixed_body) > 1e-5


def test_joint_hessian_symmetry_and_line_integral():
    r, bs, ls, _, _, m = toy()
    b, last = bs[0], ls[0]
    d, dl = r.normal(size=b.shape) / 3, r.normal(size=last.shape) / 3
    e, el = r.normal(size=b.shape) / 3, r.normal(size=last.shape) / 3

    def grad(x, l):
        return np.r_[
            hidden_gradient(c.predict(x, l, m)[1], m).ravel(), c.last_gradient(x, l, m)
        ]

    def hv(x, l):
        return (
            grad(b + 1e-5 * x, last + 1e-5 * l) - grad(b - 1e-5 * x, last - 1e-5 * l)
        ) / 2e-5

    assert np.r_[d.ravel(), dl] @ hv(e, el) == pytest.approx(
        np.r_[e.ravel(), el] @ hv(d, dl), abs=1e-8
    )
    nodes, weights = np.polynomial.legendre.leggauss(24)
    integral = sum(
        w
        / 2
        * (grad(b + (x + 1) / 2 * d, last + (x + 1) / 2 * dl) @ np.r_[d.ravel(), dl])
        for x, w in zip(nodes, weights)
    )
    assert integral == pytest.approx(
        c.scalar(b + d, last + dl, m) - c.scalar(b, last, m), abs=1e-10
    )


def test_context_reference_is_source_only_and_rank_bounded():
    _, _, ls, _, base, m = toy()
    ref = c.context_reference(ls, base)
    assert ref["context_basis"].shape == (6, 3)
    np.testing.assert_allclose(
        ref["context_basis"].T @ ref["context_basis"], np.eye(3), atol=1e-12
    )
    np.testing.assert_allclose(
        ((ls - ref["context_mean"]) @ ref["context_basis"] / ref["context_scale"]).std(
            0
        ),
        1,
    )
    np.testing.assert_array_equal(ref["context_centers"], m["context_centers"])
    with pytest.raises(ValueError, match="Rank-zero"):
        c.context_reference(np.ones_like(ls), base)


def test_equal_author_equal_view_and_direct_objective():
    _, bs, ls, gs, base, _ = toy()
    bodies = [x for b in bs for x in (b, b)]
    lasts = [x for i, l in enumerate(ls) for x in (l, np.delete(ls, i, 0).mean(0))]
    grads = [x for g in gs for x in (g, 1.1 * g)]
    ids = np.repeat(np.arange(4), 2)
    ys = np.linspace(-1, 2, 8)
    models, diag = c.fit_pair(bodies, lasts, grads, ys, ids, ls, base)
    assert diag[1]["source_objective"] <= diag[0]["source_objective"] + 1e-10
    np.testing.assert_allclose(
        c.view_weights([0, 0, 1, 1, 1]), [0.25, 0.25, 1 / 6, 1 / 6, 1 / 6]
    )
    for model, d in zip(models, diag):
        pred = [c.predict(b, l, model) for b, l in zip(bodies, lasts)]
        score_loss = (
            np.mean([(p[0] - y) ** 2 for p, y in zip(pred, ys)])
            / base["score_scale"] ** 2
        )
        field_loss = (
            np.mean(
                [
                    np.mean((p[1] - len(b) * (g @ base["basis"]) * base["scale"]) ** 2)
                    for p, b, g in zip(pred, bodies, grads)
                ]
            )
            / base["density_scale"] ** 2
        )
        assert d["source_score_loss"] == pytest.approx(score_loss, abs=1e-10)
        assert d["source_density_loss"] == pytest.approx(field_loss, abs=1e-10)
        for key in (
            "basis",
            "center",
            "scale",
            "landmarks",
            "bandwidth",
            "score_scale",
            "density_scale",
        ):
            np.testing.assert_array_equal(model[key], base[key])
    additive = models[0]
    np.testing.assert_array_equal(
        c.predict(bs[0], ls[0], additive)[1], c.predict(bs[0], ls[1], additive)[1]
    )
