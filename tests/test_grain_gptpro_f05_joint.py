"""Independent full Gaussian and causal-prefix checks for joint state filters."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f05 import eta_from_rho
from grain_research.gptpro_foundation_v3.f05_joint import (
    joint_filter,
    learn_common_variance,
)
from grain_research.gptpro_foundation_v3.f05_recovery import filter_arrays


def test_joint_matches_full_gaussian_terminal_posterior():
    psi = np.array([[0.2, 0.3, eta_from_rho(0.4)], [-0.1, 0.5, eta_from_rho(0.7)]])
    rho = 0.99 * np.tanh(psi[:, 2])
    q = np.array([[0.25, 0.125], [0.125, 0.25]])
    p = q / (1 - rho[:, None] * rho[None, :])
    y = np.random.default_rng(42).normal(size=(5, 2))
    z = np.zeros_like(y)
    blocks = []
    for i in range(5):
        blocks.append(
            [
                rho[:, None] ** (i - j) * p if i >= j else p * rho[None, :] ** (j - i)
                for j in range(5)
            ]
        )
    latent = np.block(blocks)
    covariance = latent + 0.25 * np.eye(10)
    cross = latent[-2:]
    residual = (y - psi[:, 0]).ravel()
    expected_mean = cross @ np.linalg.solve(covariance, residual)
    expected_var = p - cross @ np.linalg.solve(covariance, cross.T)
    actual = joint_filter(y, z, psi, 0.25, 0.25, 0.125)
    np.testing.assert_allclose(actual["state_mean"][-1], expected_mean, atol=1e-12)
    np.testing.assert_allclose(
        actual["state_variance"][-1], np.diag(expected_var), atol=1e-12
    )


def test_independent_limit_and_future_poisoning():
    psi = np.array([[0, 0.5, eta_from_rho(0.4)], [0, 0.5, eta_from_rho(0.7)]])
    y = np.random.default_rng(8).normal(size=(16, 2))
    z = np.zeros_like(y)
    actual = joint_filter(y, z, psi, 0.25, 0.25, 0.0)
    for i in range(2):
        expected = filter_arrays(
            psi[i], y[:, i], z[:, i], np.arange(16), np.log([0.5, 0.5])
        )
        np.testing.assert_allclose(
            actual["state_mean"][:, i], expected["state_mean"], atol=1e-12
        )
    baseline = joint_filter(y, z, psi, 0.25, 0.25, 0.125)
    changed = y.copy()
    changed[8:] += 1000
    poisoned = joint_filter(changed, z, psi, 0.25, 0.25, 0.125)
    np.testing.assert_array_equal(
        baseline["state_mean"][:8], poisoned["state_mean"][:8]
    )


def test_source_common_moment_matches_covariance():
    rng = np.random.default_rng(17)
    y = rng.normal(size=(4, 20))
    rows = [
        {
            "author_id": str(i),
            "t": t,
            "event_id": f"{i}:{t}",
            "z": 0.0,
            "y": float(y[i, t]),
            "role": "source",
        }
        for i in range(4)
        for t in range(20)
    ]
    model = {
        "log_sigma": np.log([0.5, 0.5]).tolist(),
        "rounds": [
            {"personal_fits": {str(i): {"psi": [0.0, 0.0, 0.0]} for i in range(4)}}
        ],
    }
    result = learn_common_variance(rows, model)
    covariance = np.cov(y[:, 1:])
    expected = (covariance.sum() - np.trace(covariance)) / 12
    assert result["raw_common_variance"] == pytest.approx(expected)
    with pytest.raises(ValueError):
        learn_common_variance([{**r, "role": "application"} for r in rows], model)
