#!/usr/bin/env python3
"""Geometry, compatible derivatives, source ownership and the unchanged control."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from grain_research.vector_mechanisms import context_modulated_potential as prior
from grain_research.vector_mechanisms import full_context_geometry as f
from grain_research.vector_mechanisms import hermite_token_potential as h
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    source_descriptor,
)


def toy():
    """Four unequal-length source authors with matched original/shifted targets."""
    r = np.random.default_rng(650)
    bs = [r.normal(size=(t, 7)) for t in (12, 16, 20, 24)]
    gs = [r.normal(size=b.shape) / len(b) for b in bs]
    ls = r.normal(size=(4, 7)) * np.arange(1, 8)
    base = source_descriptor(bs, gs, [0, 1, 2, 3], rank=2, landmarks=2)
    base |= {
        "coef": r.normal(size=29),
        "score_scale": np.array(1.2),
        "density_scale": np.array(0.9),
        "alpha": np.array(0.01),
        "origin": np.array(0.1),
    }
    full = (
        base
        | f.source_context(ls, base)
        | {"coef": r.normal(size=53), "arm": np.array(f.ARM)}
    )
    return r, bs, ls, gs, base, full


def test_zero_extra_nests_additive_and_reference_zero():
    _, bs, ls, _, base, m = toy()
    nested = dict(m, coef=np.r_[base["coef"], np.zeros(24)])
    for a, b in zip(f.predict(bs[0], ls[0], nested), h.predict(bs[0], ls[0], base)):
        np.testing.assert_allclose(a, b, atol=1e-12)
    short = dict(base, coef=m["coef"][:29])
    for a, b in zip(
        f.predict(bs[0], m["context_mean"], m),
        h.predict(bs[0], m["context_mean"], short),
    ):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_body_last_mixed_derivatives_and_line_integral():
    r, bs, ls, _, _, m = toy()
    b, last = bs[0], ls[0]
    d, dl = r.normal(size=b.shape), r.normal(size=last.shape)

    def grad(x, l):
        return np.r_[
            hidden_gradient(f.predict(x, l, m)[1], m).ravel(), f.last_gradient(x, l, m)
        ]

    fd = (
        f.scalar(b + 1e-5 * d, last + 1e-5 * dl, m)
        - f.scalar(b - 1e-5 * d, last - 1e-5 * dl, m)
    ) / 2e-5
    assert fd == pytest.approx(grad(b, last) @ np.r_[d.ravel(), dl], abs=1e-8)
    mixed1 = (
        np.sum(
            (
                hidden_gradient(f.predict(b, last + 1e-5 * dl, m)[1], m)
                - hidden_gradient(f.predict(b, last - 1e-5 * dl, m)[1], m)
            )
            * d
        )
        / 2e-5
    )
    mixed2 = (
        (
            f.last_gradient(b + 1e-5 * d, last, m)
            - f.last_gradient(b - 1e-5 * d, last, m)
        )
        @ dl
        / 2e-5
    )
    assert abs(mixed1) > 1e-7
    assert mixed1 == pytest.approx(mixed2, abs=1e-8)
    nodes, w = np.polynomial.legendre.leggauss(24)
    integral = sum(
        wi
        / 2
        * (grad(b + (t + 1) / 2 * d, last + (t + 1) / 2 * dl) @ np.r_[d.ravel(), dl])
        for t, wi in zip(nodes, w)
    )
    assert integral == pytest.approx(
        f.scalar(b + d, last + dl, m) - f.scalar(b, last, m), abs=1e-10
    )


def test_kernel_rotation_translation_and_offspan_factorization():
    r, _, ls, _, base, m = toy()
    last = r.normal(size=7)
    q = np.linalg.qr(r.normal(size=(7, 7)))[0]
    shift = r.normal(size=7)
    transformed = base | f.source_context(ls @ q + shift, base)
    g, d = f.context_terms(last, m)
    g2, d2 = f.context_terms(last @ q + shift, transformed)
    np.testing.assert_allclose(g, g2, atol=1e-12)
    np.testing.assert_allclose(d @ q, d2, atol=1e-12)
    info = f.geometry_account(last, m)
    assert info["factorization_error"] < 1e-12
    assert 0 < info["radial_factor"] < 1
    assert np.std(pdist(ls)) > 0.01


def test_same_data_prefix_matches_and_scalar_objective_reconstructs():
    _, bs, ls, gs, base, _ = toy()
    bodies = [b for b in bs for _ in (0, 1)]
    lasts = [x for i, l in enumerate(ls) for x in (l, np.delete(ls, i, 0).mean(0))]
    gradients = [x for g in gs for x in (g, g * 1.15)]
    ids = np.repeat(np.arange(4), 2)
    y = np.linspace(-1, 2, 8)
    controls, _ = prior.fit_pair(bodies, lasts, gradients, y, ids, ls, base)
    model, diag = f.fit(bodies, lasts, gradients, y, ids, ls, controls[0])
    assert diag["additive_normal_block_error"] < 1e-10
    assert diag["source_objective"] <= diag["frozen_additive_objective"] + 1e-10
    values = []
    errors = []
    for b, l, g, yy in zip(bodies, lasts, gradients, y):
        value, k = f.predict(b, l, model)
        values.append((value - yy) ** 2)
        errors.append(
            np.mean((k - len(b) * (g @ model["basis"]) * model["scale"]) ** 2)
        )
    assert diag["source_score_loss"] == pytest.approx(
        np.mean(values) / base["score_scale"] ** 2, abs=1e-10
    )
    assert diag["source_density_loss"] == pytest.approx(
        np.mean(errors) / base["density_scale"] ** 2, abs=1e-10
    )
    wrong = dict(controls[0], rhs=controls[0]["rhs"] + 1)
    with pytest.raises(ValueError, match="normal equations"):
        f.fit(bodies, lasts, gradients, y, ids, ls, wrong)
