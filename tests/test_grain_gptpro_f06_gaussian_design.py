"""Analytic gradients, state-parameter uncertainty and target-design checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f06_gaussian_design import (
    CANDIDATES,
    TARGETS,
    expected_variance_gain,
    fit_prefix,
    gaussian_terms,
    joint_posterior_at_map,
    objective_gradient,
    policy_action,
    source_prior,
)
from grain_research.gptpro_foundation_v3.f06_policy_state import (
    gaussian_policy_filter,
    observed_arrays,
    simulate_policy_pair,
)


def source():
    """Known fixed toy source prior, not a fitted performance claim."""
    return {
        "prior": {
            "mu": [0, 0.5, np.arctanh(0.5 / 0.99)],
            "Sigma": np.diag([1, 0.25, 0.4]).tolist(),
        },
        "log_sigma": np.log([0.5, 0.5]).tolist(),
    }


def observations():
    rng = np.random.default_rng(19)
    return simulate_policy_pair(
        0.2,
        0.4,
        0.5,
        0.3,
        0.1,
        rng.normal(0, 0.5, 16),
        rng.normal(0, 0.5, 16),
        rng.random(16),
        "hidden",
    )["observed"]


def test_map_gradient_matches_central_differences_including_zero_rho():
    rows = observations()
    y, z, h = observed_arrays(rows)
    mu, covariance, (q, r) = source_prior(source())
    precision = np.linalg.inv(covariance)
    for eta in (0.0, -0.3, 0.7):
        psi = np.array([0.3, 0.4, eta, 0.2])
        _, gradient = objective_gradient(psi, y, z, h, q, r, mu, precision)
        numeric = []
        for j in range(4):
            delta = np.eye(4)[j] * 1e-5
            numeric.append(
                (
                    objective_gradient(psi + delta, y, z, h, q, r, mu, precision)[0]
                    - objective_gradient(psi - delta, y, z, h, q, r, mu, precision)[0]
                )
                / 2e-5
            )
        np.testing.assert_allclose(gradient, numeric, atol=1e-7, rtol=1e-6)


def test_conditional_state_and_gradient_match_scalar_filter():
    rows = observations()
    y, z, h = observed_arrays(rows)
    mu, covariance, (q, r) = source_prior(source())
    posterior = joint_posterior_at_map(mu, y, z, h, q, r, np.linalg.inv(covariance))
    exact = gaussian_policy_filter(
        rows, mu[0], mu[1], 0.99 * np.tanh(mu[2]), mu[3], q, r
    )
    np.testing.assert_allclose(
        posterior["mean"][4], exact["state_mean"][-1], atol=1e-12
    )
    np.testing.assert_allclose(
        posterior["conditional_state_variance"], exact["state_variance"][-1], atol=1e-12
    )
    numeric = []
    for j in range(4):
        delta = np.eye(4)[j] * 1e-5
        plus, minus = mu + delta, mu - delta
        high = gaussian_policy_filter(
            rows, plus[0], plus[1], 0.99 * np.tanh(plus[2]), plus[3], q, r
        )["state_mean"][-1]
        low = gaussian_policy_filter(
            rows, minus[0], minus[1], 0.99 * np.tanh(minus[2]), minus[3], q, r
        )["state_mean"][-1]
        numeric.append((high - low) / 2e-5)
    np.testing.assert_allclose(posterior["state_gradient"], numeric, atol=1e-8)
    assert np.linalg.eigvalsh(posterior["covariance"]).min() > 0
    assert posterior["covariance"][4, 4] >= posterior["conditional_state_variance"]


def test_exact_conditional_covariance_update_and_candidate_selection():
    posterior = fit_prefix(observations(), source())
    assert posterior["success"]
    for target, index in TARGETS.items():
        gains = []
        for z, h in CANDIDATES:
            result = expected_variance_gain(posterior, z, h, target)
            prior = result["predictive"]
            x = result["observation"]
            joint = np.block(
                [
                    [prior, (prior @ x)[:, None]],
                    [
                        (x @ prior)[None, :],
                        np.array([[x @ prior @ x + posterior["r"]]]),
                    ],
                ]
            )
            conditioned = joint[:5, :5] - joint[:5, 5:] @ np.linalg.solve(
                joint[5:, 5:], joint[5:, :5]
            )
            np.testing.assert_allclose(conditioned, result["updated"], atol=1e-12)
            np.testing.assert_allclose(
                prior[index, index] - conditioned[index, index],
                result["gain"],
                atol=1e-12,
            )
            assert result["gain"] >= 0
            gains.append(result["gain"])
        chosen = policy_action(posterior, "scalar_information", 16, target)
        assert gains[CANDIDATES.index(chosen)] >= max(gains) - 1e-14


def test_zero_action_beta_is_prior_only():
    rows = [{**row, "h": 0} for row in observations()]
    fit = fit_prefix(rows, source())
    assert fit["beta_status"] == "prior_only"
    np.testing.assert_allclose(fit["mean"][3], 0.4, atol=1e-10)
    np.testing.assert_allclose(fit["covariance"][3, 3], 0.04, atol=1e-10)


def test_dense_likelihood_matches_sequential_gaussian():
    rows = observations()
    y, z, h = observed_arrays(rows)
    mu, covariance, (q, r) = source_prior(source())
    objective, _ = objective_gradient(mu, y, z, h, q, r, mu, np.linalg.inv(covariance))
    sequential = gaussian_policy_filter(
        rows, mu[0], mu[1], 0.99 * np.tanh(mu[2]), mu[3], q, r
    )
    np.testing.assert_allclose(-objective, sequential["loglik"], atol=1e-12)
    terms = gaussian_terms(mu, z, h, q, r)
    assert np.linalg.eigvalsh(terms["covariance"]).min() > 0
