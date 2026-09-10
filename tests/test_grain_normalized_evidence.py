#!/usr/bin/env python3
"""Matched mass coordinates, scalar derivatives and source-only evidence fitting."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import coupled_pooling as old
from grain_research.vector_mechanisms import full_context_geometry as base
from grain_research.vector_mechanisms import normalized_evidence as f
from grain_research.vector_mechanisms.token_potential import hidden_gradient
from tests.test_grain_nonlinear_pooling import toy


def test_mass_invertibility_shift_and_stable_weights():
    rng = np.random.default_rng(700)
    z, logw = rng.normal(size=(30, 3)), rng.normal(size=(30, 4)) - 3
    a, b = f.weighted_summary(z, logw), f.weighted_summary(z, logw + 5)
    np.testing.assert_allclose(a["m"], a["mass"][:, None] * a["r"])
    np.testing.assert_allclose(a["r"], a["m"] / np.exp(a["ell"])[:, None])
    np.testing.assert_allclose(b["r"], a["r"], atol=1e-14)
    np.testing.assert_allclose(b["ell"], a["ell"] + 5)
    np.testing.assert_allclose(b["m"], a["m"] * np.exp(5))
    np.testing.assert_allclose(a["m"], np.exp(logw).T @ z / len(z))
    np.testing.assert_allclose(a["a"].sum(0), 1)
    with pytest.raises(FloatingPointError, match="no clipping"):
        f.weighted_summary(z, logw - 1000)
    with pytest.raises(ValueError, match="Finite"):
        f.weighted_summary(z, logw * np.nan)


@pytest.mark.parametrize("arm", f.ARMS)
def test_full_scalar_density_last_gradient_and_source_center(arm):
    rng, records, local = toy()
    m = f.reference(records, local, arm)
    extra = len(m["summary_center"])
    assert extra == 16 * (2 + 1)
    m["coef"] = np.r_[local["coef"], rng.normal(size=extra)]
    b, last = records[0]["body"], records[0]["last"]
    d, dl = rng.normal(size=b.shape), rng.normal(size=last.shape)
    fd = (
        f.scalar(b + 1e-5 * d, last + 1e-5 * dl, m)
        - f.scalar(b - 1e-5 * d, last - 1e-5 * dl, m)
    ) / 2e-5
    analytic = (
        np.sum(hidden_gradient(f.predict(b, last, m)[1], m) * d)
        + f.last_gradient(b, last, m) @ dl
    )
    assert fd == pytest.approx(analytic, abs=1e-8)
    nested = dict(m, coef=np.r_[local["coef"], np.zeros(extra)])
    for x, y in zip(
        f.predict(b, last, nested), base.predict(b, last, local), strict=True
    ):
        np.testing.assert_allclose(x, y, atol=1e-12)
    center = m["summary_center"].copy()
    f.predict(b + 1, last, m)
    np.testing.assert_array_equal(center, m["summary_center"])
    original = [r for r in records if r["view"] == "original"]
    expected = [
        f.summary_features(*base.design(r["body"], r["last"], m)[:2], m)
        for r in original
    ]
    np.testing.assert_allclose(center, np.mean(expected, axis=0))


@pytest.mark.parametrize("arm", f.ARMS)
def test_mixed_derivative_symmetry_and_logmass_coupling(arm):
    rng, records, local = toy()
    m = f.reference(records, local, arm)
    # Isolate logmass: even the unnormalized control has cross-token response.
    m["coef"] = np.r_[np.zeros(len(local["coef"]) + 32), rng.normal(size=16)]
    b, last = records[0]["body"], records[0]["last"]
    a, c = np.zeros_like(b), np.zeros_like(b)
    a[:20], c[20:] = rng.normal(size=a[:20].shape), rng.normal(size=c[20:].shape)
    grad = lambda x: hidden_gradient(f.predict(x, last, m)[1], m)
    ab = np.sum((grad(b + 1e-5 * a) - grad(b - 1e-5 * a)) * c) / 2e-5
    ba = np.sum((grad(b + 1e-5 * c) - grad(b - 1e-5 * c)) * a) / 2e-5
    assert abs(ab) > 1e-9
    assert ab == pytest.approx(ba, abs=1e-8)


def test_two_fits_preserve_prefix_and_stationary_objective():
    _, records, m = toy()
    local = old.fit(records, m)[0][0]
    models, diags = f.fit(records, local)
    assert len(models) == 2
    for model, diag in zip(models, diags, strict=True):
        assert len(model["coef"]) == 437
        assert diag["matched_prefix_error"] < 1e-9
        assert diag["objective_reconstruction_error"] < 1e-9
        assert diag["normal_equation_error"] < 1e-9
        scales = old.energies(records, model)
        d = np.linspace(-0.1, 0.1, len(model["coef"]))
        other = dict(model, coef=model["coef"] + 0.001 * d)
        penalty = np.eye(len(d)) * 0.01
        penalty[0, 0] = 0
        delta = (
            f.objective(records, other, scales).sum()
            - f.objective(records, model, scales).sum()
        )
        assert delta == pytest.approx(
            1e-6 * d @ (model["gram"] + penalty) @ d, abs=1e-8
        )
    with pytest.raises(ValueError, match="normalization"):
        f.fit(records, dict(local, source_cross_energy=np.array(10.0)))


def test_source_layout_and_kernel_contract():
    _, records, m = toy()
    with pytest.raises(ValueError, match="four unique"):
        f.reference(records[:-1], m, f.ARMS[0])
    with pytest.raises(ValueError, match="64 frozen"):
        f.reference(records, dict(m, landmarks=m["landmarks"][:32]), f.ARMS[0])
