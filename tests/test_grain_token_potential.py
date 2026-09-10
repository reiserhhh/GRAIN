#!/usr/bin/env python3
"""Synthetic NEXT60 scalar, gradient, weighting and source-reference contracts."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.token_potential import (
    ARMS,
    equal_author_moments,
    fit_potential,
    hidden_gradient,
    last_gradient,
    predict_potential,
    source_descriptor,
    validate_members,
)


def toy(arm):
    """Use non-unit scaling and an orthonormal non-axis-aligned source projection."""
    r = np.random.default_rng(601)
    h = r.normal(size=(7, 5))
    last = r.normal(size=5)
    u = np.linalg.qr(r.normal(size=(5, 3)))[0]
    m = {
        "basis": u,
        "center": np.array([0.2, -0.3, 0.5]),
        "scale": np.array([2.0, 0.7, 1.3]),
        "landmarks": r.normal(size=(6, 6)),
        "bandwidth": np.array(1.4),
        "coef": r.normal(size=13),
        "members": np.array([0, 2, 3]),
        "arm": np.array(arm),
    }
    return r, h, last, m


@pytest.mark.parametrize("arm", ARMS)
def test_exact_hidden_and_last_gradients(arm):
    r, h, last, m = toy(arm)
    _, k = predict_potential(h, last, m)
    g = hidden_gradient(k, m)
    dh, dl = r.normal(size=h.shape), r.normal(size=last.shape)
    eps = 1e-5
    yp, ym = [
        predict_potential(h + s * eps * dh, last + s * eps * dl, m)[0] for s in (1, -1)
    ]
    expected = np.sum(g * dh) + last_gradient(m) @ dl
    assert (yp - ym) / (2 * eps) == pytest.approx(expected, abs=2e-9)


def test_zero_mean_allocation_not_erased_before_nonlinearity():
    values = []
    for arm in ARMS:
        _, h, last, m = toy(arm)
        _, k = predict_potential(h, last, m)
        g = hidden_gradient(k, m)
        d = h - h.mean(axis=0)
        values.append(np.sum(g * d))
    assert abs(values[0]) < 1e-14
    assert abs(values[1]) > 1e-4


def test_equal_author_weight_not_token_weight():
    a = np.array([[1.0, 2], [3, 4]])
    b = np.array([[10.0, -1], [12, 2], [14, 5]])
    mean, sd = equal_author_moments([a, b])
    np.testing.assert_allclose(mean, (a.mean(0) + b.mean(0)) / 2)
    other = equal_author_moments([np.repeat(a, 9, axis=0), b])
    np.testing.assert_allclose(other, (mean, sd))
    assert not np.allclose(mean, np.concatenate([a, b]).mean(0))


@pytest.mark.parametrize("arm", ARMS)
def test_same_scalar_line_integral(arm):
    r, h, last, m = toy(arm)
    d, dl = r.normal(size=h.shape) / 3, r.normal(size=last.shape) / 3
    x, weights = np.polynomial.legendre.leggauss(32)
    integral = 0.0
    for t, w in zip((x + 1) / 2, weights / 2):
        _, k = predict_potential(h + t * d, last + t * dl, m)
        integral += w * (np.sum(hidden_gradient(k, m) * d) + last_gradient(m) @ dl)
    delta = predict_potential(h + d, last + dl, m)[0] - predict_potential(h, last, m)[0]
    assert integral == pytest.approx(delta, abs=1e-12)


@pytest.mark.parametrize("arm", ARMS)
def test_fit_and_target_independent_application(arm):
    r = np.random.default_rng(603)
    hs = [r.normal(size=(t, 5)) for t in (18, 25, 32)]
    gs = [r.normal(size=h.shape) / len(h) for h in hs]
    lasts = [r.normal(size=5) for _ in hs]
    desc = source_descriptor(hs, gs, [0, 2, 3], rank=3, landmarks=4)
    model, diag = fit_potential(hs, lasts, gs, [1, -1, 0.5], desc, arm)
    validate_members(model, [0, 2, 3])
    with pytest.raises(ValueError, match="exclusion"):
        validate_members(model, [0, 1, 2])
    h, last = r.normal(size=(19, 5)), r.normal(size=5)
    before = predict_potential(h, last, model)
    gs[0][:] = 1000
    after = predict_potential(h, last, model)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert diag["normal_equation_error"] < 1e-8
    assert np.all(
        desc["basis"][np.argmax(abs(desc["basis"]), axis=0), np.arange(3)] > 0
    )
    np.testing.assert_allclose(desc["basis"].T @ desc["basis"], np.eye(3), atol=1e-12)


def test_descriptor_rejects_duplicate_source_members():
    r = np.random.default_rng(604)
    hs = [r.normal(size=(18, 5)) for _ in range(2)]
    with pytest.raises(ValueError, match="membership"):
        source_descriptor(hs, hs, [1, 1], rank=3)
