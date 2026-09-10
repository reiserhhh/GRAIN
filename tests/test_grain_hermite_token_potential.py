#!/usr/bin/env python3
"""Hermite scalar compatibility, exact nesting, source objective and QR tests."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.hermite_token_potential import (
    density_blocks,
    fit,
    local_features,
    predict,
    qr_capacity,
)
from grain_research.vector_mechanisms.token_potential import (
    fit_potential,
    hidden_gradient,
    last_gradient,
    predict_potential,
    source_descriptor,
    validate_members,
)


def toy():
    """Build a small source-scaled descriptor with several directional centers."""
    r = np.random.default_rng(620)
    hs = [r.normal(size=(t, 5)) for t in (16, 22, 29)]
    gs = [r.normal(size=h.shape) / len(h) for h in hs]
    lasts = [r.normal(size=5) for _ in hs]
    y = np.array([-1.0, 0.5, 2.0])
    desc = source_descriptor(hs, gs, [0, 2, 3], rank=3, landmarks=4)
    base, _ = fit_potential(hs, lasts, gs, y, desc, "kernel_then_mean")
    m = dict(base)
    m["coef"] = r.normal(size=len(base["coef"]) + 3 * len(base["landmarks"]))
    return r, hs, lasts, gs, y, base, m


def test_zero_extra_coefficients_recover_old_function():
    _, hs, lasts, _, _, base, m = toy()
    m["coef"] = np.r_[base["coef"], np.zeros(len(m["coef"]) - len(base["coef"]))]
    old, new = predict_potential(hs[0], lasts[0], base), predict(hs[0], lasts[0], m)
    for a, b in zip(old, new):
        np.testing.assert_allclose(a, b, atol=1e-13, rtol=1e-13)


def test_body_and_last_finite_differences():
    r, hs, lasts, _, _, _, m = toy()
    h, last = hs[0], lasts[0]
    d, dl = r.normal(size=h.shape), r.normal(size=last.shape)
    _, k = predict(h, last, m)
    exact = np.sum(hidden_gradient(k, m) * d) + last_gradient(m) @ dl
    yp, ym = [predict(h + s * 1e-5 * d, last + s * 1e-5 * dl, m)[0] for s in (1, -1)]
    assert (yp - ym) / 2e-5 == pytest.approx(exact, abs=1e-8)


def test_hessian_bilinear_symmetry():
    r, hs, lasts, _, _, _, m = toy()
    h, last = hs[0], lasts[0]
    d, e = r.normal(size=h.shape), r.normal(size=h.shape)

    def hv(x):
        gp, gm = [
            hidden_gradient(predict(h + s * 1e-5 * x, last, m)[1], m) for s in (1, -1)
        ]
        return (gp - gm) / 2e-5

    assert np.sum(d * hv(e)) == pytest.approx(np.sum(e * hv(d)), abs=1e-8)


def test_scalar_line_integral():
    r, hs, lasts, _, _, _, m = toy()
    h, last = hs[0], lasts[0]
    d, dl = r.normal(size=h.shape) / 4, r.normal(size=last.shape) / 4
    x, w = np.polynomial.legendre.leggauss(32)
    integral = 0.0
    for t, weight in zip((x + 1) / 2, w / 2):
        _, k = predict(h + t * d, last + t * dl, m)
        integral += weight * (np.sum(hidden_gradient(k, m) * d) + last_gradient(m) @ dl)
    assert integral == pytest.approx(
        predict(h + d, last + dl, m)[0] - predict(h, last, m)[0], abs=1e-11
    )


def test_center_has_independent_directional_derivatives():
    m = {"landmarks": np.zeros((1, 5)), "bandwidth": np.array(2.0)}
    z, p = np.zeros((2, 2)), np.zeros((2, 3))
    _, density = next(density_blocks(z, p, m))
    np.testing.assert_allclose(
        density[:, :, 6:], np.broadcast_to(np.eye(2) / 2, (2, 2, 2))
    )
    np.testing.assert_array_equal(local_features(z, p, m)[1], np.zeros((2, 1, 2)))


def test_source_fit_nested_objective_and_exclusion():
    r, hs, lasts, gs, y, base, _ = toy()
    model, diag = fit(hs, lasts, gs, y, base)
    assert diag["source_objective"] <= diag["baseline_source_objective"] + 1e-10
    validate_members(model, [0, 2, 3])
    with pytest.raises(ValueError, match="exclusion"):
        validate_members(model, [0, 1, 3])
    h, last = r.normal(size=(10, 5)), r.normal(size=5)
    before = predict(h, last, model)
    gs[0][:] = 999
    after = predict(h, last, model)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_qr_projection_matches_direct_svd():
    r = np.random.default_rng(621)
    d = r.normal(size=(60, 12))
    d[:, -1] = d[:, 0] + d[:, 1]
    k = r.normal(size=60)
    result, _ = qr_capacity(d, k)
    theta = np.linalg.lstsq(d, k, rcond=1e-10)[0]
    assert result["source_field_capacity_loss"] == pytest.approx(
        np.sum((d @ theta - k) ** 2), abs=1e-11
    )
    assert result["source_field_rank"] == 11
    assert result["capacity_energy_error"] < 1e-10


def test_coordinate_chunking_does_not_change_design():
    r, _, _, _, _, _, m = toy()
    z, p = r.normal(size=(31, 3)), r.normal(size=(31, 3))
    a = np.concatenate([d for _, d in density_blocks(z, p, m, chunk_size=3)])
    b = np.concatenate([d for _, d in density_blocks(z, p, m, chunk_size=128)])
    np.testing.assert_array_equal(a, b)


def test_equal_author_objective_matches_direct_reconstruction():
    _, hs, lasts, gs, y, base, _ = toy()
    model, diag = fit(hs, lasts, gs, y, base)
    values, losses = [], []
    for h, last, g in zip(hs, lasts, gs, strict=True):
        value, k = predict(h, last, model)
        target = len(h) * (g @ model["basis"]) * model["scale"]
        values.append(value)
        losses.append(np.mean((k - target) ** 2))
    assert diag["source_score_loss"] == pytest.approx(
        np.mean((np.array(values) - y) ** 2) / model["score_scale"] ** 2,
        abs=1e-10,
    )
    assert diag["source_field_loss"] == pytest.approx(
        np.mean(losses) / model["density_scale"] ** 2, abs=1e-10
    )
