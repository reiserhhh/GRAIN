#!/usr/bin/env python3
"""Observable summaries can collide while target-active joint structure differs."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_natural_summary as M
from scripts.run_grain_math_natural_summary import raw_moments


def support():
    return np.asarray(list(itertools.product((-1, 1), repeat=4)), float)


def test_equal_mean_std_retains_active_cross_product():
    g = support()
    y = (g[:, 0] * g[:, 1])[:, None]
    r = M.witness(M.design(g, np.zeros(len(g)), "pooled_diag"), y)
    assert r["gap"][0] == pytest.approx(1)
    assert r["residual_energy_fraction"][0] == pytest.approx(1)
    pp, pm = r["plus"][:, 0], r["minus"][:, 0]
    np.testing.assert_allclose(pp @ g, pm @ g, atol=1e-14)
    np.testing.assert_allclose(pp @ (g**2), pm @ (g**2), atol=1e-14)
    assert min(pp.min(), pm.min()) > 0


def test_full_covariance_retains_second_but_not_third_order():
    g = support()
    y = np.column_stack((g[:, 0] * g[:, 1], np.prod(g[:, :3], axis=1)))
    r = M.witness(M.design(g, np.zeros(len(g)), "pooled_full"), y)
    assert not r["active"][0]
    assert r["active"][1]


def test_full_second_moment_span_is_rotation_invariant():
    rng = np.random.default_rng(10)
    g, y = rng.normal(size=(40, 4)), rng.normal(size=(40, 2))
    panel = np.arange(40) % 3
    a = M.witness(M.design(g, panel, "pooled_full"), y)
    b = M.witness(M.design(g, panel, "rotated_full"), y)
    np.testing.assert_allclose(a["residual"], b["residual"], atol=1e-12)
    np.testing.assert_allclose(M.ROTATION @ M.ROTATION, np.eye(4), atol=1e-14)


def test_diagonal_summary_is_not_rotation_invariant():
    g = support()
    y = (g[:, 0] * g[:, 1] + g[:, 2] * g[:, 3])[:, None]
    a = M.witness(M.design(g, np.zeros(16), "pooled_diag"), y)
    b = M.witness(M.design(g, np.zeros(16), "rotated_diag"), y)
    assert a["residual_energy_fraction"][0] == pytest.approx(1)
    assert b["residual_energy_fraction"][0] < 1e-20


def test_full_embedding_mean_on_finite_independent_support():
    g, h, y = support(), np.eye(16), np.arange(16)[:, None]
    r = M.witness(M.design(g, np.zeros(16), "embedding_mean", h), y)
    assert r["rank"] == 16
    assert not r["active"].any()


def test_panels_preserve_distinct_weights_and_conditioned_means():
    rng = np.random.default_rng(44)
    g = rng.normal(size=(30, 4))
    panel = np.arange(30) % 3
    y = rng.normal(size=(30, 1))
    r = M.witness(M.design(g, panel, "panel_diag"), y)
    d = (r["plus"] - r["minus"])[:, 0]
    for i in np.unique(panel):
        np.testing.assert_allclose(d[panel == i].sum(), 0, atol=1e-12)
        np.testing.assert_allclose(d[panel == i] @ g[panel == i], 0, atol=1e-12)


@pytest.mark.parametrize("radius", [0, 1, -1])
def test_invalid_radius_fails(radius):
    with pytest.raises(ValueError):
        M.witness(np.ones((1, 3)), np.zeros((3, 1)), radius)


def test_constant_target_is_not_reported_as_signal():
    r = M.witness(np.ones((1, 5)), np.ones((5, 1)))
    assert not r["active"].any()
    assert r["gap"][0] == 0


@pytest.mark.parametrize("arm", M.ARMS)
def test_original_moment_witness_audit(arm):
    rng = np.random.default_rng(105)
    g = rng.normal(size=(40, 4)) * np.arange(1, 5) + np.arange(4)
    h, y = rng.normal(size=(40, 7)), rng.normal(size=(40, 2))
    panel = np.arange(40) % 3
    result = M.witness(M.design(g, panel, arm, h), y)
    for j in range(2):
        a = raw_moments(g, panel, result["plus"][:, j], arm, h)
        b = raw_moments(g, panel, result["minus"][:, j], arm, h)
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(
            result["gap"][j],
            result["plus"][:, j] @ y[:, j] - result["minus"][:, j] @ y[:, j],
            atol=1e-12,
        )


def test_mean_embedding_retains_any_fixed_linear_query_even_without_full_rank():
    rng = np.random.default_rng(111)
    g, h = rng.normal(size=(40, 4)), rng.normal(size=(40, 3))
    y = h @ rng.normal(size=(3, 2)) + 7
    result = M.witness(M.design(g, np.arange(40) % 3, "embedding_mean", h), y)
    assert result["null_dimension"] > 0
    assert not result["active"].any()


def test_empty_query_or_malformed_design_fails():
    with pytest.raises(ValueError):
        M.witness(np.ones(5), np.ones((5, 1)))
    with pytest.raises(ValueError):
        M.witness(np.ones((1, 5)), np.ones((5, 0)))
