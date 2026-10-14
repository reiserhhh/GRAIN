#!/usr/bin/env python3
"""Verify exact conditional Gaussian elimination against dense conditioning."""

import numpy as np

from grain_research.gptpro_foundation_v3.f16_dynamic_gaussian_reference import (
    conditional_filter,
    joint_future,
)


def test_augmented_filter_equals_dense_conditional_gaussian():
    model = {
        "prior": {
            "mu": [0.1, 0.4, 0.5],
            "Sigma": [[1.0, 0.1, 0.05], [0.1, 0.4, -0.02], [0.05, -0.02, 0.1]],
        },
        "log_sigma": np.log([0.5, 0.4]),
    }
    eta = np.array([0.1, 0.7])
    z, y = np.array([[1.0, -1, 1, -1]]), np.array([[0.4, 0.2, -0.5, 0.7]])
    weights, mask = np.array([[1.0, 0.5, 0, 1.0]]), np.array([[1, 1, 0, 1]], bool)
    m, c, ll = conditional_filter(model, eta, (z, y, weights, mask))
    sig = np.array(model["prior"]["Sigma"])
    mu = np.array(model["prior"]["mu"])
    for i, e in enumerate(eta):
        rho = 0.99 * np.tanh(e)
        times = np.array([0, 1, 3])
        prior_mu = mu[:2] + sig[:2, 2] / sig[2, 2] * (e - mu[2])
        prior_cov = sig[:2, :2] - np.outer(sig[:2, 2], sig[2, :2]) / sig[2, 2]
        h = np.column_stack([np.ones(3), z[0, times]])
        state = 0.25 / (1 - rho * rho)
        v = (
            h @ prior_cov @ h.T
            + state * rho ** np.abs(times[:, None] - times[None, :])
            + np.diag(0.16 / weights[0, times])
        )
        cross = np.vstack([prior_cov @ h.T, state * rho ** (3 - times)])
        target_cov = np.zeros((3, 3))
        target_cov[:2, :2] = prior_cov
        target_cov[2, 2] = state
        resid = y[0, times] - h @ prior_mu
        expected = np.r_[prior_mu, 0] + cross @ np.linalg.solve(v, resid)
        expected_cov = target_cov - cross @ np.linalg.solve(v, cross.T)
        np.testing.assert_allclose(m[i, 0], expected, atol=1e-12)
        np.testing.assert_allclose(c[i, 0], expected_cov, atol=1e-12)
        expected_ll = -0.5 * (
            3 * np.log(2 * np.pi)
            + np.linalg.slogdet(v)[1]
            + resid @ np.linalg.solve(v, resid)
        )
        np.testing.assert_allclose(ll[i, 0], expected_ll, atol=1e-12)
    future, state, parameters = joint_future(m, c, eta, model["log_sigma"])
    np.testing.assert_allclose(future[:, :, 0], m[:, :, 2])
    np.testing.assert_allclose((state + parameters)[:, :, 0, 0], c[:, :, 2, 2])
    assert np.min(np.linalg.eigvalsh(state + parameters)) > 0
