"""Check GLS profile objectives and exact prior curvature addition."""

import numpy as np
from scipy.optimize import minimize

from grain_research.gptpro_foundation_v3.f05_profiles import (
    central_hessian,
    profile_point,
)
from grain_research.gptpro_foundation_v3.f05_recovery import (
    generate_dynamics,
    kalman_loglik,
)


def test_profile_matches_numerical_nuisance_optimization():
    rows, _ = generate_dynamics(
        "core", 23, {"source": 1, "validation": 0, "application": 0}, 32
    )
    logs = np.log([0.5, 0.4])
    prior = {
        "mu": [0.0, 0.5, 0.7],
        "Sigma": [[1.0, 0.1, 0.2], [0.1, 0.4, 0.05], [0.2, 0.05, 0.5]],
    }
    for selected in (None, prior):
        result = profile_point(rows, 0.6, logs, selected)

        def objective(beta, result=result, selected=selected):
            psi = np.r_[beta, result["psi"][2]]
            loss = -kalman_loglik(psi, rows, logs)["loglik"]
            if selected:
                delta = psi - np.asarray(prior["mu"])
                loss += 0.5 * delta @ np.linalg.inv(prior["Sigma"]) @ delta
            return loss

        numerical = minimize(objective, [0.0, 0.0], method="BFGS")
        np.testing.assert_allclose(result["psi"][:2], numerical.x, atol=1e-5)
        np.testing.assert_allclose(
            result["objective"], objective(result["psi"][:2]), atol=1e-10
        )


def test_hessian_quadratic_and_prior_addition():
    p = np.array([[2.0, 0.1, 0.0], [0.1, 3.0, 0.2], [0.0, 0.2, 1.0]])
    f = lambda x: 0.5 * x @ p @ x
    point = np.array([0.2, -0.3, 0.7])
    np.testing.assert_allclose(central_hessian(f, point), p, atol=1e-7)
    prior = np.diag([1.0, 2.0, 3.0])
    g = lambda x: f(x) + 0.5 * x @ prior @ x
    np.testing.assert_allclose(
        central_hessian(g, point) - central_hessian(f, point), prior, atol=1e-7
    )
