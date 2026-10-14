#!/usr/bin/env python3
"""Joint future covariance checked against stationary dense conditioning."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f05_recovery import dense_posterior
from grain_research.gptpro_foundation_v3.f13_dynamic import (
    joint_forecast,
    target_projection,
)


@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.6, 0.97])
def test_dense_joint_conditioning(rho):
    rng = np.random.default_rng(133)
    times = np.array([0, 1, 4, 6, 7])
    z = np.array([-1, 1, 1, -1, 1])
    y = rng.normal(size=5)
    psi = np.array([0.3, 0.7, np.arctanh(rho / 0.99)])
    logs = np.log([0.5, 0.4])
    rows = [
        {"event_id": str(i), "t": int(t), "z": float(zz), "y": float(yy)}
        for i, (t, zz, yy) in enumerate(zip(times, z, y, strict=True))
    ]
    filtered = dense_posterior(psi, rows, logs)
    future_z = np.array([-1.0, 1.0, -1.0, 1.0])
    mean, covariance, _, _ = joint_forecast(
        psi, logs, filtered["state_mean"], filtered["state_variance"], future_z
    )
    target_times = times[-1] + np.arange(5)
    stationary = 0.25 / (1 - rho * rho)
    observed = stationary * rho ** np.abs(times[:, None] - times) + 0.16 * np.eye(5)
    cross = stationary * rho ** np.abs(target_times[:, None] - times)
    target = stationary * rho ** np.abs(target_times[:, None] - target_times) + np.diag(
        [0, 0.16, 0.16, 0.16, 0.16]
    )
    expected_mean = np.r_[0.0, psi[0] + psi[1] * future_z] + cross @ np.linalg.solve(
        observed, y - psi[0] - psi[1] * z
    )
    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(
        covariance, target - cross @ np.linalg.solve(observed, cross.T), atol=1e-12
    )


def test_average_requires_cross_time_covariances():
    _, cov, _, _ = joint_forecast(
        [0, 0, np.arctanh(0.8 / 0.99)], np.log([0.5, 0.5]), 0, 0.2, [0] * 4
    )
    w = target_projection()[2]
    expected = 0.2 * (sum(0.8**k for k in range(1, 5)) / 4) ** 2
    expected += (
        0.25 / 16 * sum(sum(0.8**l for l in range(4 - j + 1)) ** 2 for j in range(1, 5))
        + 0.25 / 4
    )
    np.testing.assert_allclose(w @ cov @ w, expected)
    assert w @ cov @ w > w @ np.diag(np.diag(cov)) @ w


def test_invalid_variance_rejected():
    with pytest.raises(ValueError):
        joint_forecast([0, 0, 0], [0, 0], 0, -1, [0] * 4)
