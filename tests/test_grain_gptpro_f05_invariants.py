"""Numerical laws checked against direct state simulation algebra."""

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from grain_research.gptpro_foundation_v3.f05_invariants import (
    conditional_future,
    covariance_jacobian,
    observation_covariance,
    similarity_transform,
    stationary_covariance,
)


def model():
    return (
        np.array([[0.6, 0.15], [0.0, 0.3]]),
        np.diag([0.25, 0.16]),
        np.array([[1.0, 1.0]]),
        0.25,
    )


def test_stationary_covariance_independent_solver():
    a, q, _, _ = model()
    np.testing.assert_allclose(
        stationary_covariance(a, q), solve_discrete_lyapunov(a, q), atol=1e-12
    )


@pytest.mark.parametrize(
    "s",
    [
        np.diag([2.0, 0.5]),
        np.array(
            [
                [np.cos(np.pi / 5), -np.sin(np.pi / 5)],
                [np.sin(np.pi / 5), np.cos(np.pi / 5)],
            ]
        ),
    ],
)
def test_full_law_and_forecast_similarity(s):
    a, q, c, r = model()
    p = stationary_covariance(a, q)
    aa, qq, cc, pp = similarity_transform(a, q, c, p, s)
    np.testing.assert_allclose(stationary_covariance(aa, qq), pp, atol=1e-12)
    times = np.array([0, 1, 4, 9, 10, 17])
    np.testing.assert_allclose(
        observation_covariance(a, q, c, r, times),
        observation_covariance(aa, qq, cc, r, times),
        atol=1e-12,
    )
    y = np.random.default_rng(42).normal(size=24)
    for expected, actual in zip(
        conditional_future(a, q, c, r, y), conditional_future(aa, qq, cc, r, y)
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_restriction_changes_parameter_space():
    a, q, c, r = model()
    unrestricted = covariance_jacobian(a, q, c, r)
    restricted = covariance_jacobian(a, q, c, r, free="fixed_CQR")
    assert unrestricted["parameters"] == 10
    assert restricted["parameters"] == 4
    assert unrestricted["rank"] < unrestricted["parameters"]
    assert np.isfinite(restricted["jacobian"]).all()


def test_unit_root_intercept_initial_shift():
    times = np.arange(64)
    a, state0, shift = 0.3, -0.2, 1.7
    np.testing.assert_allclose(
        a + state0 * np.ones(64),
        (a + shift) + (state0 - shift) * np.ones(64),
        atol=1e-12,
    )
    rho = 0.98
    change = (a + shift) + (state0 - shift) * rho**times - (a + state0 * rho**times)
    np.testing.assert_allclose(change, shift * (1 - rho**times), atol=1e-12)
    assert abs(change[-1]) > 1


def test_noise_covariance_ambiguity_is_not_only_similarity():
    a, q, c, r = model()
    v = np.array([1.0, -1.0]) / np.sqrt(2)
    delta_p = 0.02 * np.outer(v, v)
    alternative_q = q + delta_p - a @ delta_p @ a.T
    assert np.linalg.eigvalsh(alternative_q).min() > 0
    assert np.linalg.matrix_rank(np.vstack((c, c @ a))) == 2
    assert np.linalg.norm(alternative_q - q) > 0.01
    np.testing.assert_allclose(
        observation_covariance(a, q, c, r, np.arange(64)),
        observation_covariance(a, alternative_q, c, r, np.arange(64)),
        atol=1e-12,
    )
