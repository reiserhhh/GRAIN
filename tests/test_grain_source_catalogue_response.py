#!/usr/bin/env python3
"""Finite null and alternative worlds for source-catalogue response diagnostics."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_source_catalogue_response as M


def world():
    """Four authors per profile, two-window positive kernels on two actual edits."""
    g = np.repeat(np.arange(32), 4)
    f = np.tile((0, 0, 1, 1), 32)
    k = np.exp(np.random.default_rng(216).normal(size=(1, 2, 2, 2, len(g))) / 5)
    return k, g, f


def test_orbit_mean_by_full_enumeration():
    """Average all six allocations in one cell; product covariance is not erased."""
    k, g, f = world()
    groups = M.strata(g)
    expected = M.orbit_mean(k, g, f, groups)
    actual = []
    for positions in itertools.combinations(range(4), 2):
        ff = f.copy()
        ff[:4] = 1
        ff[list(positions)] = 0
        actual.append(M.laws(k, g, ff)[..., 0])
    np.testing.assert_allclose(np.mean(actual, 0), expected[..., 0], atol=1e-14)
    # Mean empirical products differ from products of the expected marginals.
    a = k[..., :4].mean(-1).prod(-1)
    assert np.max(abs(expected[0, :, :, 2, :, 0] - a)) > 1e-4


def test_exact_condition_support_and_exchange():
    """A subgroup can freeze an entire cell without declaring equal response."""
    k, g, f = world()
    c = np.tile(np.array(((0, 0), (0, 0), (12, 12), (12, 12))), (32, 1))
    assert M.support(g, f)["movable_authors"] == 128
    assert M.support(g, f, c)["movable_authors"] == 0
    groups = M.strata(g, c)
    ff = M.exchange(f, groups, np.random.default_rng(1))
    np.testing.assert_array_equal(ff, f)
    np.testing.assert_allclose(M.orbit_mean(k, g, f, groups), M.laws(k, g, f))
    assert np.max(abs(M.response(M.laws(k, g, f))["conditional_cross_ratio"])) > 0.01


def test_unequal_bank_product_expectations():
    """The same pool gives equal mean single/joint laws, not equal mean products."""
    k, g, _ = world()
    f = np.tile((0, 1, 1, 1), 32)
    expected = M.orbit_mean(k, g, f, M.strata(g))
    actual = []
    for i in range(4):
        ff = f.copy()
        ff[:4] = 1
        ff[i] = 0
        actual.append(M.laws(k, g, ff)[..., 0])
    np.testing.assert_allclose(np.mean(actual, 0), expected[..., 0], atol=1e-14)
    np.testing.assert_allclose(expected[0, :, :, :2], expected[1, :, :, :2])
    assert np.max(abs(expected[0, :, :, 2] - expected[1, :, :, 2])) > 1e-4


def test_full_group_preserves_profiles_and_conditions():
    """Neither labels nor vector-condition relations are permuted."""
    _, g, f = world()
    c = np.tile(np.array(((3, 4), (3, 4), (3, 4), (3, 4))), (32, 1))
    groups = M.strata(g, c)
    ff = M.exchange(f, groups, np.random.default_rng(2))
    for ix in groups:
        np.testing.assert_array_equal(np.sort(f[ix]), np.sort(ff[ix]))
    assert np.any(ff != f)


def test_identical_catalogue_null():
    """Target-dependent kernels can coexist with exactly zero bank-response drift."""
    k, g, f = world()
    for code in range(32):
        k[..., g == code] = k[..., g == code].mean(-1, keepdims=True)
    r = M.response(M.laws(k, g, f))
    np.testing.assert_allclose(r["conditional_cross_ratio"], 0, atol=1e-14)
    np.testing.assert_allclose(r["response_defect"], 0, atol=1e-14)
    assert np.max(abs(r["source_action"])) > 0.001


def test_gaussian_components_and_shapes():
    """Fixed kernels equal literal distances, not a new high-precision estimator."""
    rng = np.random.default_rng(3)
    p, z = rng.normal(size=(3, 2, 2, 2, 4)), rng.normal(size=(64, 2, 4))
    k = M.kernels(p, z, 2.0)
    for u, v, a, w, i in np.ndindex(3, 2, 2, 2, 64):
        assert (
            abs(k[u, v, a, w, i] - np.exp(-np.sum((p[u, v, a, w] - z[i, w]) ** 2) / 4))
            < 1e-14
        )
    g, f = np.repeat(np.arange(32), 2), np.tile((0, 1), 32)
    r = M.response(M.laws(k, g, f))
    assert M.statistics(r)["cross_ratio_RMS"].shape == (3, 5)
    assert M.statistics(r)["all5_cross_ratio_RMS"].shape == (3,)


@pytest.mark.parametrize("bad", [np.zeros((64, 2)) + 0.5, np.ones((64, 2)) * 13])
def test_invalid_condition_counts(bad):
    """Fractional or out-of-protocol conditions cannot fabricate overlap."""
    with pytest.raises(ValueError):
        M.strata(np.repeat(np.arange(32), 2), bad)


def test_missing_profile_rejected():
    """No unsupported target profile is repaired with neutral density."""
    with pytest.raises(ValueError):
        M.support(np.arange(32), np.zeros(32, int))
