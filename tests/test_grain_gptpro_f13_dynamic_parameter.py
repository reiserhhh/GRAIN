#!/usr/bin/env python3
"""Parameter-mixture validation without truth-assisted personal estimation."""

import numpy as np

from grain_research.gptpro_foundation_v3.f05_recovery import dense_posterior
from grain_research.gptpro_foundation_v3.f13_dynamic import (
    joint_forecast,
    target_projection,
)
from grain_research.gptpro_foundation_v3.f13_dynamic_parameter import (
    conditional_target_components,
    filter_parameter_batch,
    laplace_person,
    parameter_mixture,
)


def test_batched_parameters_against_dense_and_joint_law():
    rng = np.random.default_rng(135)
    p = rng.normal(size=(20, 3))
    rows = [
        {"event_id": str(i), "t": t, "y": float(rng.normal()), "z": (-1.0) ** i}
        for i, t in enumerate([0, 1, 3, 4, 8, 11])
    ]
    logs = np.log([0.5, 0.4])
    means, variances = filter_parameter_batch(p, rows, logs)
    target_means, target_cov = conditional_target_components(
        p, rows, logs, [-1, 1, 1, -1]
    )
    projection = target_projection()
    for i, psi in enumerate(p):
        oracle = dense_posterior(psi, rows, logs)
        np.testing.assert_allclose(means[i], oracle["state_mean"], atol=1e-12)
        np.testing.assert_allclose(variances[i], oracle["state_variance"], atol=1e-12)
        m, v, _, _ = joint_forecast(psi, logs, means[i], variances[i], [-1, 1, 1, -1])
        np.testing.assert_allclose(target_means[i], projection @ m, atol=1e-12)
        np.testing.assert_allclose(
            target_cov[i], projection @ v @ projection.T, atol=1e-12
        )
    mixed = parameter_mixture(p, rows, logs, [-1, 1, 1, -1])
    expected = (target_cov + target_means[:, :, None] * target_means[:, None, :]).mean(
        axis=0
    ) - np.outer(mixed["mean"], mixed["mean"])
    np.testing.assert_allclose(mixed["total"], expected, atol=1e-12)
    assert np.linalg.eigvalsh(mixed["parameter"]).min() >= -1e-12


def test_laplace_curvature_matches_dense_likelihood():
    """Hessian is checked using dense observations, not the production filter."""
    from grain_research.gptpro_foundation_v3.f05_profiles import central_hessian

    rng = np.random.default_rng(139)
    rows = [
        {"event_id": str(i), "t": i, "z": float(i % 2), "y": float(rng.normal())}
        for i in range(24)
    ]
    model = {
        "log_sigma": np.log([0.5, 0.5]).tolist(),
        "prior": {"mu": [0, 0.5, 0.4], "Sigma": np.diag([1.0, 0.25, 1.0]).tolist()},
    }
    result = laplace_person(rows, model)
    assert result["status"] == "LAPLACE_APPROXIMATION"
    mu = np.array(model["prior"]["mu"])
    precision = np.linalg.inv(model["prior"]["Sigma"])

    def dense_objective(p):
        d = p - mu
        return (
            -dense_posterior(p, rows, model["log_sigma"])["loglik"]
            + 0.5 * d @ precision @ d
        )

    expected = central_hessian(dense_objective, result["fit"]["psi"])
    np.testing.assert_allclose(result["hessian"], expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(result["hessian"]) @ result["parameter_covariance"],
        np.eye(3),
        atol=1e-12,
    )
