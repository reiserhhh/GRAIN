#!/usr/bin/env python3
"""Paired teaching, scalar compatibility, source weights and exact objective checks."""

import numpy as np
import pytest
from test_grain_full_context_geometry import toy

from grain_research.vector_mechanisms import context_modulated_potential as c
from grain_research.vector_mechanisms import full_context_geometry as f
from grain_research.vector_mechanisms import paired_response_objective as p
from grain_research.vector_mechanisms.token_potential import hidden_gradient


def source():
    """Construct a tiny source-only fit with two unequal-length views per author."""
    r, bs, ls, gs, base, _ = toy()
    b = [x for x in bs for _ in (0, 1)]
    l = [x for i, x0 in enumerate(ls) for x in (x0, np.delete(ls, i, 0).mean(0))]
    g = [x for g0 in gs for x in (g0, g0 * 1.15)]
    y, ids = np.linspace(-1, 2, 8), np.repeat(np.arange(4), 2)
    controls, _ = c.fit_pair(b, l, g, y, ids, ls, base)
    model, _ = f.fit(b, l, g, y, ids, ls, controls[0])
    return r, b, l, g, y, ids, model


def test_equal_author_energy_and_pair_membership():
    x = [np.ones((2, 3)), np.ones((40, 3)) * 3]
    assert p.delta_energy(x, 1) == pytest.approx(5)
    assert p.delta_energy([v * 7 for v in x], 7) == pytest.approx(245)
    with pytest.raises(ValueError, match="Unsupported"):
        p.delta_energy([np.zeros((4, 2))], 1)
    with pytest.raises(ValueError, match="Finite"):
        p.delta_energy([np.array([np.nan])], 1)
    with pytest.raises(ValueError, match="two views"):
        p.paired_indices([0, 0, 1], [0, 1])
    with pytest.raises(ValueError, match="membership"):
        p.paired_indices([0, 0, 1, 1], [0])


def test_normal_update_exchange_and_independent_losses():
    _, b, l, g, y, ids, m = source()
    dg, dr, e, diag = p.normals(b, l, g, y, ids, m)
    assert diag["base_normal_error"] < 1e-10
    assert diag["additive_delta_prefix_error"] == 0
    assert np.linalg.eigvalsh(dg).min() > -1e-10
    exchange = np.arange(8).reshape(4, 2)[:, ::-1].ravel()
    dg2, dr2, e2, _ = p.normals(
        *[[v[i] for i in exchange] for v in (b, l, g, y, ids)], m
    )
    np.testing.assert_allclose(dg, dg2, atol=1e-12)
    np.testing.assert_allclose(dr, dr2, atol=1e-12)
    assert e == e2
    new, out = p.fit(b, l, g, y, ids, m)
    np.testing.assert_allclose(new["gram"], m["gram"] + dg)
    np.testing.assert_allclose(new["rhs"], m["rhs"] + dr)
    assert out["source_objective"] <= out["frozen_augmented_objective"] + 1e-10
    assert out["objective_reconstruction_error"] < 1e-10
    loss = p.objective(b, l, g, y, ids, new, e)
    np.testing.assert_allclose(
        loss,
        [
            out[k]
            for k in (
                "source_score_loss",
                "source_density_loss",
                "source_delta_loss",
                "source_penalty",
            )
        ],
    )
    bad = dict(m, rhs=m["rhs"] + 1)
    with pytest.raises(ValueError, match="base normal"):
        p.normals(b, l, g, y, ids, bad)


def test_trained_scalar_body_last_and_mixed_derivatives():
    r, b, l, g, y, ids, m = source()
    model, _ = p.fit(b, l, g, y, ids, m)
    d, dl = r.normal(size=b[0].shape), r.normal(size=l[0].shape)
    eps = 1e-5
    fd = (
        f.scalar(b[0] + eps * d, l[0] + eps * dl, model)
        - f.scalar(b[0] - eps * d, l[0] - eps * dl, model)
    ) / (2 * eps)
    actual = np.sum(hidden_gradient(f.predict(b[0], l[0], model)[1], model) * d)
    actual += f.last_gradient(b[0], l[0], model) @ dl
    assert fd == pytest.approx(actual, abs=1e-8)
    mixed1 = np.sum(
        (
            hidden_gradient(f.predict(b[0], l[0] + eps * dl, model)[1], model)
            - hidden_gradient(f.predict(b[0], l[0] - eps * dl, model)[1], model)
        )
        * d
    ) / (2 * eps)
    mixed2 = (
        (
            f.last_gradient(b[0] + eps * d, l[0], model)
            - f.last_gradient(b[0] - eps * d, l[0], model)
        )
        @ dl
        / (2 * eps)
    )
    assert mixed1 == pytest.approx(mixed2, abs=1e-8)


def test_small_delta_loss_is_not_the_endpoint_average_objective():
    # A scalar toy parameter controls a small response; fixed regularization can
    # suppress it under endpoint loss, while explicit relative teaching retains it.
    eps, alpha = 0.001, 0.01
    old = eps**2 / (eps**2 + alpha)
    new = (eps**2 + 1) / (eps**2 + 1 + alpha)
    assert old < 0.001 and new > 0.99
    assert (1 - new) ** 2 < (1 - old) ** 2
    assert alpha * new**2 > alpha * old**2
