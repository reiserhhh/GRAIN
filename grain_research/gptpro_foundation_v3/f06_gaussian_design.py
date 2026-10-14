"""MAP-local Gaussian observation design with correctly lagged intervention."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from grain_research.gptpro_foundation_v3.f06_policy_state import observed_arrays

CANDIDATES = tuple(product((-1.0, 1.0), (0.0, 1.0)))
TARGETS = {"a": 0, "b": 1, "beta": 3, "state": 4}


def source_prior(source):
    """Extend frozen F05 prior by the protocol's independent beta prior."""
    mean = np.r_[source["prior"]["mu"], 0.4]
    covariance = np.zeros((4, 4))
    covariance[:3, :3] = source["prior"]["Sigma"]
    covariance[3, 3] = 0.04
    np.linalg.cholesky(covariance)
    return mean, covariance, np.exp(2 * np.asarray(source["log_sigma"]))


def gaussian_terms(psi, z, h, q, r):
    """Mean/covariance and exact first derivatives for the realized action path."""
    a, b, eta, beta = psi
    rho = 0.99 * np.tanh(eta)
    drho = 0.99 * (1 - np.tanh(eta) ** 2)
    n = len(z)
    forcing, derivative = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        forcing[t] = rho * forcing[t - 1] + h[t - 1]
        derivative[t] = forcing[t - 1] + rho * derivative[t - 1]
    lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    denominator = 1 - rho**2
    power = rho**lag
    derivative_power = np.zeros_like(power)
    nonzero = lag > 0
    derivative_power[nonzero] = lag[nonzero] * rho ** (lag[nonzero] - 1)
    covariance = q * power / denominator + np.eye(n) * r
    covariance_eta = (
        q * (derivative_power / denominator + 2 * rho * power / denominator**2) * drho
    )
    mean = a + b * z + beta * forcing
    mean_derivatives = np.column_stack(
        [np.ones(n), z, beta * derivative * drho, forcing]
    )
    return {
        "mean": mean,
        "covariance": covariance,
        "covariance_eta": covariance_eta,
        "mean_derivatives": mean_derivatives,
        "forcing": forcing,
        "forcing_eta": derivative * drho,
        "rho": rho,
        "drho": drho,
    }


def objective_gradient(psi, y, z, h, q, r, prior_mean, precision):
    """Exact Gaussian MAP objective and analytic gradient, not finite differences."""
    terms = gaussian_terms(psi, z, h, q, r)
    factor = cho_factor(terms["covariance"], lower=True)
    residual = y - terms["mean"]
    solved = cho_solve(factor, residual)
    inverse = cho_solve(factor, np.eye(len(y)))
    deviation = psi - prior_mean
    objective = 0.5 * (
        len(y) * np.log(2 * np.pi)
        + 2 * np.log(np.diag(factor[0])).sum()
        + residual @ solved
        + deviation @ precision @ deviation
    )
    gradient = -terms["mean_derivatives"].T @ solved + precision @ deviation
    gradient[2] += 0.5 * np.sum(
        (inverse - np.outer(solved, solved)) * terms["covariance_eta"]
    )
    return float(objective), gradient


def joint_posterior_at_map(psi, y, z, h, q, r, prior_precision):
    """Expected-information Gaussian approximation plus conditional state uncertainty.

    The covariance is inverse expected information at MAP, not an exact posterior
    or observed-Hessian Laplace claim. State-parameter covariance is retained.
    """
    terms = gaussian_terms(psi, z, h, q, r)
    factor = cho_factor(terms["covariance"], lower=True)
    solved_derivatives = cho_solve(factor, terms["mean_derivatives"])
    fisher = terms["mean_derivatives"].T @ solved_derivatives
    covariance_action = cho_solve(factor, terms["covariance_eta"])
    fisher[2, 2] += 0.5 * np.trace(covariance_action @ covariance_action)
    parameter_covariance = np.linalg.inv(prior_precision + fisher)
    rho, drho = terms["rho"], terms["drho"]
    lag = len(y) - 1 - np.arange(len(y))
    stationary = q / (1 - rho**2)
    cross = stationary * rho**lag
    cross_derivative = 2 * q * rho / (1 - rho**2) ** 2 * rho**lag
    active = lag > 0
    cross_derivative[active] += stationary * lag[active] * rho ** (lag[active] - 1)
    cross_derivative *= drho
    residual_solve = cho_solve(factor, y - terms["mean"])
    cross_solve = cho_solve(factor, cross)
    state_mean = psi[3] * terms["forcing"][-1] + cross @ residual_solve
    state_variance = stationary - cross @ cross_solve
    state_gradient = -terms["mean_derivatives"].T @ cross_solve
    state_gradient[3] += terms["forcing"][-1]
    state_gradient[2] += (
        psi[3] * terms["forcing_eta"][-1]
        + cross_derivative @ residual_solve
        - cross_solve @ terms["covariance_eta"] @ residual_solve
    )
    joint = np.zeros((5, 5))
    joint[:4, :4] = parameter_covariance
    joint[:4, 4] = parameter_covariance @ state_gradient
    joint[4, :4] = joint[:4, 4]
    joint[4, 4] = (
        state_variance + state_gradient @ parameter_covariance @ state_gradient
    )
    return {
        "mean": np.r_[psi, state_mean],
        "covariance": joint,
        "state_gradient": state_gradient,
        "conditional_state_variance": float(state_variance),
        "information": fisher,
        "rho": rho,
        "drho": drho,
    }


def fit_prefix(rows, source):
    """Refit current prefix only, two frozen starts, explicit fallback on failure."""
    prior_mean, prior_covariance, (q, r) = source_prior(source)
    precision = np.linalg.inv(prior_covariance)
    if not rows:
        rho = 0.99 * np.tanh(prior_mean[2])
        joint = np.zeros((5, 5))
        joint[:4, :4] = prior_covariance
        joint[4, 4] = q / (1 - rho**2)
        return {
            "mean": np.r_[prior_mean, 0.0],
            "covariance": joint,
            "rho": rho,
            "drho": 0.99 * (1 - np.tanh(prior_mean[2]) ** 2),
            "success": True,
            "attempts": [],
            "q": q,
            "r": r,
            "beta_status": "prior_only",
            "mode": "source_prior_before_first_observation",
        }
    y, z, h = observed_arrays(rows)
    slope = np.linalg.lstsq(np.column_stack([np.ones(len(y)), z]), y, rcond=None)[0][1]
    attempts, candidates = [], []
    for start in (0.2, 0.8):
        fit = minimize(
            objective_gradient,
            [y.mean(), slope, np.arctanh(start / 0.99), 0.4],
            args=(y, z, h, q, r, prior_mean, precision),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-6},
        )
        attempts.append(
            {
                "start_rho": start,
                "success": bool(fit.success),
                "iterations": int(fit.nit),
                "objective": float(fit.fun),
                "message": str(fit.message),
            }
        )
        if fit.success and np.isfinite(fit.fun) and np.isfinite(fit.x).all():
            candidates.append(fit)
    psi = min(candidates, key=lambda f: f.fun).x if candidates else prior_mean
    posterior = joint_posterior_at_map(psi, y, z, h, q, r, precision)
    if not candidates:
        # Keep the declared source covariance rather than a failed fit's certainty.
        replacement = np.zeros((5, 5))
        replacement[:4, :4] = prior_covariance
        g = posterior["state_gradient"]
        replacement[:4, 4] = prior_covariance @ g
        replacement[4, :4] = replacement[:4, 4]
        replacement[4, 4] = (
            posterior["conditional_state_variance"] + g @ prior_covariance @ g
        )
        posterior["covariance"] = replacement
    return {
        **posterior,
        "success": bool(candidates),
        "attempts": attempts,
        "q": q,
        "r": r,
        "beta_status": "prior_only" if not np.any(h[:-1]) else "prefix_estimated",
        "mode": "MAP_expected_information" if candidates else "source_only_fallback",
    }


def expected_variance_gain(posterior, condition, intervention, target):
    """Predict next state, then condition on next response for one scalar target."""
    if target not in TARGETS or (condition, intervention) not in CANDIDATES:
        raise ValueError("unknown target or candidate")
    mean = posterior["mean"]
    transition = np.eye(5)
    transition[4] = [0, 0, posterior["drho"] * mean[4], intervention, posterior["rho"]]
    predictive = transition @ posterior["covariance"] @ transition.T
    predictive[4, 4] += posterior["q"]
    observation = np.array([1.0, condition, 0.0, 0.0, 1.0])
    cross = predictive @ observation
    variance = observation @ cross + posterior["r"]
    updated = predictive - np.outer(cross, cross) / variance
    index = TARGETS[target]
    return {
        "gain": float(cross[index] ** 2 / variance),
        "before": float(predictive[index, index]),
        "after": float(updated[index, index]),
        "predictive": predictive,
        "updated": updated,
        "transition": transition,
        "observation": observation,
    }


def policy_action(posterior, policy, next_t, target="a", random_index=0):
    """Select (next z,current h); realized y and latent state are not arguments."""
    if policy == "fixed":
        return (-1.0 if next_t % 2 == 0 else 1.0, 0.0)
    if policy == "random":
        return CANDIDATES[int(random_index)]
    if policy == "coverage":
        return CANDIDATES[next_t % 4]
    if policy != "scalar_information":
        raise ValueError("unknown policy")
    gains = np.array(
        [expected_variance_gain(posterior, z, h, target)["gain"] for z, h in CANDIDATES]
    )
    maximum = gains.max()
    return CANDIDATES[
        int(np.flatnonzero(np.isclose(gains, maximum, atol=1e-14, rtol=0))[0])
    ]
