#!/usr/bin/env python3
"""Synthetic controls for distribution-based measurement, not validity claims."""

import numpy as np
import pytest

from grain_research.construction import math_author_moment_measurement as M


def test_cube_full_moments_reconstruct_distribution():
    _, _, w = M.cube(3)
    p = np.arange(1, 9, dtype=float) / 36
    np.testing.assert_allclose(w @ M.moments(p, 3) / 8, p)


def test_contraction_vector_and_scalar():
    rng = np.random.default_rng(62)
    p = rng.dirichlet(np.ones(8))
    f = rng.normal(size=(8, 4))
    np.testing.assert_allclose(
        M.integrate(f, p, 3), M.moments(p, 3) @ M.coefficients(f, 3)
    )
    assert M.integrate(f[:, 0], p, 3) == pytest.approx(M.integrate(f, p, 3)[0])


def test_pairwise_equal_but_third_order_changes_reading():
    x, _, _ = M.cube(3)
    pp, pm = M.parity_profiles([1, 1, 1], dimension=3)
    a = M.moment_design(2, 3)
    np.testing.assert_allclose(a @ pp, a @ pm)
    assert M.integrate(np.prod(x, axis=1), pp, 3) - M.integrate(
        np.prod(x, axis=1), pm, 3
    ) == pytest.approx(1)


def test_additive_first_moment_interval_collapses():
    x, _, _ = M.cube(3)
    f = 0.4 + x @ np.array([0.1, 0.2, -0.1])
    mu = M.moment_design(1, 3) @ np.full(8, 1 / 8)
    interval = M.sharp_interval(f, mu, 1, 3)
    assert interval["width"] == pytest.approx(0, abs=1e-10)
    assert interval["lower"]["value"] == pytest.approx(0.4)


def test_pure_third_order_sharp_bounds_and_full_collapse():
    x, _, _ = M.cube(3)
    f = np.prod(x, axis=1)
    for order in (1, 2):
        mu = M.moment_design(order, 3) @ np.full(8, 1 / 8)
        interval = M.sharp_interval(f, mu, order, 3)
        assert interval["lower"]["value"] == pytest.approx(-1)
        assert interval["upper"]["value"] == pytest.approx(1)
        for side in ("lower", "upper"):
            assert interval[side]["error"] < 1e-10
    mu = M.moment_design(3, 3) @ np.full(8, 1 / 8)
    assert M.sharp_interval(f, mu, 3, 3)["width"] == pytest.approx(0, abs=1e-10)


def test_boundary_support_can_identify_nonadditive_response():
    x, _, _ = M.cube(3)
    p = np.zeros(8)
    p[-1] = 1
    mu = M.moment_design(1, 3) @ p
    assert M.sharp_interval(np.prod(x, 1), mu, 1, 3)["width"] == pytest.approx(0)


@pytest.mark.parametrize("p", [[0.5, 0.5], [0.2] * 8, [np.nan] * 8, [-1] + [2 / 7] * 7])
def test_invalid_distribution_fails(p):
    with pytest.raises(ValueError):
        M.probability(p, 3)


def test_infeasible_moments_fail_without_fake_interval():
    with pytest.raises(ValueError, match="LP"):
        M.sharp_interval(np.zeros(8), [1, 0, 0, 2], 1, 3)


def test_empty_or_nonbinary_profile_fails():
    for mask in ([0, 0, 0], [1, 2, 0], [1, 0]):
        with pytest.raises(ValueError):
            M.parity_profiles(mask, dimension=3)
