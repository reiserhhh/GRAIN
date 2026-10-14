#!/usr/bin/env python3
"""Known finite mechanisms and non-independent counterexamples."""

import itertools

import numpy as np

from grain_research.construction import math_distinct_condition_geometry as U


def as_views(rows):
    """Duplicate an adapter view, not independent observed sampling."""
    return np.repeat(np.asarray(rows)[:, None], 2, axis=1)


def test_independent_zero_world_has_no_mean_unequal_event_geometry():
    rows = np.asarray(list(itertools.product((-1.0, 1.0), repeat=4)))[..., None]
    c = as_views(np.tile([0, 0, 1, 1], (len(rows), 1)))
    m = U.observations(as_views(rows), c)
    np.testing.assert_allclose(m["geometry"][:, 0].mean(0).ravel(), [1, 0, 1])
    np.testing.assert_allclose(
        m["geometry"], U.literal_observations(as_views(rows), c)["geometry"]
    )


def test_correlated_repeat_world_is_not_unbiased_by_fiat():
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=2)))
    rows = np.repeat(signs, 2, axis=1)[..., None]
    c = as_views(np.tile([0, 0, 1, 1], (len(rows), 1)))
    g = U.observations(as_views(rows), c)["geometry"][:, 0].mean(0).ravel()
    np.testing.assert_allclose(g, [2, 2, 0])


def test_true_displacement_without_dispersion_survives():
    row = np.array([[[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [1.0, 2.0]]])
    c = as_views(np.array([[0, 0, 1, 1]]))
    m = U.observations(as_views(row), c)
    np.testing.assert_allclose(m["geometry"][..., 0, :, :], m["geometry"][..., 1, :, :])
    np.testing.assert_array_equal(m["geometry"][..., 2, :, :], 0)


def test_negative_geometry_is_retained():
    row = np.array([[[-1.0], [1.0], [-1.0], [1.0]]])
    c = as_views(np.array([[0, 0, 1, 1]]))
    g = U.observations(as_views(row), c)["geometry"][0, 0, :, 0, 0]
    np.testing.assert_allclose(g, [0, -2, 2])


def test_singleton_missing_is_not_zero():
    row = np.array([[[0.0], [1.0], [2.0], [3.0]]])
    c = as_views(np.array([[0, 0, 0, 1]]))
    m = U.observations(as_views(row), c)
    assert m["conditional_support"].all()
    assert not m["support"].any()
    assert np.isnan(m["geometry"]).all()


def test_orthogonal_coordinates_and_full_account():
    rng = np.random.default_rng(9)
    row = rng.normal(size=(6, 2, 8, 3))
    c = np.tile([0, 0, 0, 0, 1, 1, 1, 1], (6, 2, 1))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    m, transformed = U.observations(row, c), U.observations(row @ q, c)
    g = m["geometry"]
    np.testing.assert_allclose(
        g[..., 0, :, :], g[..., 1, :, :] + g[..., 2, :, :], atol=1e-14
    )
    np.testing.assert_allclose(transformed["geometry"], q.T @ g @ q, atol=1e-14)
    np.testing.assert_allclose(
        g, U.literal_observations(row, c)["geometry"], atol=1e-14
    )
