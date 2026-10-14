#!/usr/bin/env python3
"""Exact small-world checks for the observation-law to criterion interface."""

from itertools import product

import numpy as np
import pytest

from grain_research.construction import math_block_observation as M


def test_all_two_block_assignments_match_negative_covariance():
    q = np.arange(18, dtype=float).reshape(1, 6, 3) / 10
    mu, gamma = M.orbit_moments(q)
    views = np.array(
        [(q[0, :3][a] + q[0, 3:][b]) / 2 for a, b in product(M.PERMUTATIONS, repeat=2)]
    )
    centered = views - mu[0]
    np.testing.assert_allclose(views.mean(0), np.broadcast_to(mu[0], (3, 3)))
    np.testing.assert_allclose(centered[:, 0].T @ centered[:, 0] / 36, 2 * gamma[0])
    np.testing.assert_allclose(centered[:, 0].T @ centered[:, 1] / 36, -gamma[0])
    np.testing.assert_allclose(views.sum(1), np.broadcast_to(3 * mu[0], (36, 3)))


def test_event_choice_and_conditional_masses_are_not_removed():
    z = np.arange(36, dtype=float).reshape(2, 6, 3)
    c = np.tile([0, 1, 1], (2, 2))
    q = M.event_features(z, c, np.tile(np.arange(6), (2, 1)))
    np.testing.assert_array_equal(q[..., 2:5] + q[..., 5:], z)
    np.testing.assert_array_equal(q[..., 1], c)
    with pytest.raises(ValueError, match="Distinct"):
        M.event_features(z, c, np.zeros_like(c))


def test_assignment_preserves_all_event_units_and_chronological_blocks():
    q = np.arange(72, dtype=float).reshape(2, 12, 3)
    views, assignments = M.partition_views(q, 16, 254)
    assert views.shape == (2, 16, 3, 3)
    np.testing.assert_array_equal(
        np.sort(assignments, -1), np.broadcast_to([0, 1, 2], assignments.shape)
    )
    np.testing.assert_allclose(
        views.sum(2), np.broadcast_to(3 * q.mean(1)[:, None], (2, 16, 3))
    )
    np.testing.assert_array_equal(views, M.partition_views(q, 16, 254)[0])


def test_finite_corrected_equation_is_whole_bank_regression():
    rng = np.random.default_rng(4)
    q = np.concatenate([np.ones((60, 6, 1)), rng.normal(size=(60, 6, 2))], -1)
    mu, gamma = M.orbit_moments(q)
    y = rng.normal(size=(60, 2))
    p = M.construct(mu, gamma, y)
    mtx, g = np.asarray(p["M"]), np.asarray(p["G"])
    np.testing.assert_allclose((mtx - g) + g, mtx)
    np.testing.assert_allclose(
        p["operators"]["finite_bank"]["beta"],
        np.linalg.lstsq(mu, y, rcond=1e-6)[0],
        atol=1e-12,
    )


def test_negative_orbit_eigenvalue_is_retained_not_clipped():
    q = np.tile(np.array([[[1.0, -1], [1, 0], [1, 1]]]), (10, 1, 1))
    mu, gamma = M.orbit_moments(q)
    p = M.construct(mu, gamma, np.arange(10)[:, None])
    assert p["operators"]["orbit_proxy"]["eigenvalues"][0] < 0


def test_cold_routing_uses_only_other_fold_and_exact_scalar_variance():
    rng = np.random.default_rng(8)
    q = np.concatenate([np.ones((40, 6, 1)), rng.normal(size=(40, 6, 2))], -1)
    mu, gamma = M.orbit_moments(q)
    folds = np.arange(40) % 2
    views, _ = M.partition_views(q, 16, 254)
    y = rng.normal(size=(40, 2))
    programs = [
        {
            **M.construct(mu[folds == f], gamma[folds == f], y[folds == f]),
            "source_fold": f,
        }
        for f in (0, 1)
    ]
    r = M.cold_readings(mu, gamma, views, folds, programs)
    for f in (0, 1):
        held = np.flatnonzero(folds != f)
        beta = np.asarray(programs[f]["operators"]["finite_bank"]["beta"])
        np.testing.assert_allclose(r["mean_scores"][1, held], mu[held] @ beta)
        for u in held:
            for t in range(2):
                assert r["partition_variance"][1, u, t] == pytest.approx(
                    2 * beta[:, t] @ gamma[u] @ beta[:, t]
                )


def test_generating_propensity_is_not_the_finite_bank_estimand():
    p = M.generating_world()
    assert p["exhaustive_rows"] == 128
    assert p["propensity_slope"] == pytest.approx(1)
    assert p["finite_bank_slope"] == pytest.approx(6 / 7)
    assert p["single_view_slope"] == pytest.approx(2 / 3)


def test_signed_allocation_account_keeps_original_equation_residual():
    rng = np.random.default_rng(2)
    mtx, g, oldk = rng.normal(size=(3, 3, 3))
    rhs, oldrhs, beta = rng.normal(size=(3, 3, 2))
    delta_k, delta_m = oldk - (mtx - g), oldrhs - rhs
    terms = [g @ beta, delta_m, -delta_k @ beta, oldk @ beta - oldrhs]
    np.testing.assert_allclose(sum(terms), mtx @ beta - rhs, atol=1e-14)
