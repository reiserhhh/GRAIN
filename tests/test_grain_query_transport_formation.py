#!/usr/bin/env python3
"""Tests for signed query-transport accounting, exclusion and rounding bounds."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.query_transport_formation import (
    STAGES,
    UPDATE_LABELS,
    WINDOWS,
    decompose,
    moment_accounting,
    prefix_updates,
)


def test_telescoping_and_fixed_windows_retain_initial():
    rng = np.random.default_rng(45)
    i, u, w = (
        rng.normal(size=(2, 4, 7)),
        rng.normal(size=(2, 4, 17, 7)),
        rng.normal(size=(2, 7)),
    )
    d = i + u.sum(2)
    a = decompose(i, u, w, d)
    np.testing.assert_allclose(a["phi"].sum(-1), a["target"], atol=1e-13)
    np.testing.assert_allclose(
        a["query_actions"][0] - a["query_actions"][1], a["phi"], atol=1e-13
    )
    np.testing.assert_allclose(
        sum(a["phi"][..., lo:hi].sum(-1) for _, lo, hi in WINDOWS),
        a["target"],
        atol=1e-13,
    )
    assert len(STAGES) == 18 and STAGES[-1] == "block25_attention"


def test_future_updates_cannot_enter_transport():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 21, 9))
    labels = list(UPDATE_LABELS) + [
        "block25_mlp",
        "block26_attention",
        "block26_mlp",
        "block27_attention",
    ]
    p = prefix_updates(a, labels)
    a[:, 17:] = 1e20
    np.testing.assert_array_equal(prefix_updates(a, labels), p)
    with pytest.raises(ValueError, match="ordered"):
        prefix_updates(a, labels[1:] + labels[:1])


@pytest.mark.parametrize("kind", ("rounding", "bad_boundary", "bad_shape", "nonfinite"))
def test_parity_is_a_vector_check_not_only_a_scalar_match(kind):
    rng = np.random.default_rng(8)
    i, u, w = (
        np.zeros((2, 4, 8)),
        rng.normal(size=(2, 4, 17, 8)),
        rng.normal(size=(2, 8)),
    )
    d = u.sum(2)
    if kind == "rounding":
        d += rng.normal(scale=1e-7, size=d.shape)
        a = decompose(i, u, w, d)
        assert np.all(abs(a["scalar_error"]) <= a["scalar_bound"] + 1e-12)
    else:
        if kind == "bad_boundary":
            d += 1
        if kind == "bad_shape":
            u = u[:, :, :16]
        if kind == "nonfinite":
            u[0, 0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            decompose(i, u, w, d)


def test_negative_increments_and_cross_energy_are_not_noise():
    z = np.array([-2.0, -1.0, 1.0, 2.0])
    x = np.column_stack((3 * z, -2 * z, np.zeros(4)))
    a = moment_accounting(x, z)
    np.testing.assert_allclose(a["share"], [3, -2, 0])
    assert a["cross_energy"] < 0
    assert a["reconstructed_energy"] == pytest.approx(a["target_energy"])
    assert a["reconstructed_variance"] == pytest.approx(a["target_variance"])
    assert a["moment_covariance_closure"] < 1e-12
    zero = moment_accounting(np.column_stack((z, -z)), np.zeros(4))
    assert np.isnan(zero["share"]).all()
    assert zero["target_energy"] == 0


def test_equal_queries_give_zero_contrast_despite_large_updates():
    rng = np.random.default_rng(90)
    i, u, w = (
        rng.normal(size=(2, 4, 6)),
        rng.normal(size=(2, 4, 17, 6)),
        rng.normal(size=(2, 6)),
    )
    i[1], u[1] = i[0], u[0]
    a = decompose(i, u, w, i + u.sum(2))
    np.testing.assert_array_equal(a["phi"], 0)
