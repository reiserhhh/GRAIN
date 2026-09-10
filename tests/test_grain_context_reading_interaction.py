#!/usr/bin/env python3
"""Fixed BODY, source exclusion and additive versus conditional read-in witnesses."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.context_reading_interaction import (
    account,
    other_source_last,
    replace_last,
)
from grain_research.vector_mechanisms.hermite_token_potential import predict
from tests.test_grain_hermite_token_potential import toy


def test_other_source_mean_excludes_self_and_other_queries():
    bank = {(i, q): np.ones(5) * (i + 100 * q) for i in range(4) for q in (0, 1)}
    before, ids = other_source_last(bank, 2, 0, (0, 1))
    assert ids == [0, 1, 3]
    np.testing.assert_allclose(before, np.full(5, 4 / 3))
    bank[2, 0][:] = 999
    for i in range(4):
        bank[i, 1][:] = -999
    np.testing.assert_array_equal(before, other_source_last(bank, 2, 0, (0, 1))[0])


def test_replacement_only_touches_last_and_preserves_dtype():
    x = np.arange(40, dtype=np.float32).reshape(8, 5)
    y = replace_last(x, np.full(5, 1 / 3), np.array([1, 3, 5]))
    assert y.dtype == x.dtype
    np.testing.assert_array_equal(y[:-1], x[:-1])
    np.testing.assert_array_equal(x[-1], np.arange(35, 40))
    with pytest.raises(ValueError):
        replace_last(x, np.ones(5), np.array([1, 7]))


def test_additive_hermite_body_field_is_exactly_last_invariant():
    r, hs, lasts, _, _, _, model = toy()
    oldscore, oldfield = predict(hs[0], lasts[0], model)
    newscore, newfield = predict(hs[0], lasts[0] + r.normal(size=lasts[0].shape), model)
    np.testing.assert_array_equal(oldfield, newfield)
    assert abs(newscore - oldscore) > 1e-6


def test_conditional_scalar_has_nonzero_finite_mixed_derivative():
    r = np.random.default_rng(630)
    z, c, dc, d = (
        r.normal(size=(12, 3)),
        r.normal(size=3),
        r.normal(size=3),
        r.normal(size=(12, 3)),
    )

    def value(x, ctx):
        return np.mean(np.sin(x @ ctx))

    def gradient(x, ctx):
        return np.cos(x @ ctx)[:, None] * ctx[None] / len(x)

    delta = gradient(z, c + dc) - gradient(z, c)
    eps = 1e-5
    contrast = (
        value(z + eps * d, c + dc)
        - value(z - eps * d, c + dc)
        - value(z + eps * d, c)
        + value(z - eps * d, c)
    ) / (2 * eps)
    assert contrast == pytest.approx(np.sum(delta * d), abs=1e-8)
    assert np.linalg.norm(delta) > 0.01


def test_field_and_signed_action_decomposition_use_fixed_directions():
    r = np.random.default_rng(631)
    g, gp = r.normal(size=(15, 6)), r.normal(size=(15, 6))
    ds = r.normal(size=(4, 15, 6))
    saved = ds.copy()
    u = np.linalg.qr(r.normal(size=(6, 3)))[0]
    row, a = account(g, gp, ds, u)
    assert row["energy_relative_error"] < 1e-12
    assert row["action_closure_error"] < 1e-12
    np.testing.assert_allclose(a["changed_tau"] - a["tau"], a["delta_tau"], atol=1e-12)
    np.testing.assert_array_equal(saved, ds)


def test_zero_change_is_representable_without_nan():
    g, ds = np.ones((5, 3)), np.ones((4, 5, 3))
    row, a = account(g, g, ds, np.eye(3))
    assert row["field_change_ratio"] == 0
    assert all(np.isfinite(v) for v in row.values())
    np.testing.assert_array_equal(a["delta_tau"], np.zeros(4))
