#!/usr/bin/env python3
"""Rao-Blackwellized Gaussian reference: integrate a,b,state analytically given eta."""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from .f16_dynamic_text import PLAN, TARGETS, prepare, validate_bundle


def conditional_filter(model, eta, arrays):
    """Exact augmented Kalman filtering conditional on each inertia coordinate."""
    z, obs, precision, mask = arrays
    mu, sigma = np.asarray(model["prior"]["mu"]), np.asarray(model["prior"]["Sigma"])
    eta = np.asarray(eta)
    n, a = len(eta), len(z)
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.asarray(model["log_sigma"]))
    ab_mean = mu[:2] + (eta - mu[2])[:, None] * sigma[:2, 2] / sigma[2, 2]
    ab_cov = sigma[:2, :2] - np.outer(sigma[:2, 2], sigma[2, :2]) / sigma[2, 2]
    means = np.zeros((n, a, 3))
    means[:, :, :2] = ab_mean[:, None, :]
    cov = np.zeros((n, a, 3, 3))
    cov[:, :, :2, :2] = ab_cov
    cov[:, :, 2, 2] = (q / (1 - rho**2))[:, None]
    ll = np.zeros((n, a))
    for t in range(z.shape[1]):
        if t:
            means[:, :, 2] *= rho[:, None]
            cov[:, :, 2, :] *= rho[:, None, None]
            cov[:, :, :, 2] *= rho[:, None, None]
            cov[:, :, 2, 2] += q
        visible = mask[:, t] & (precision[:, t] > 0)
        if not visible.any():
            continue
        m, c = means[:, visible], cov[:, visible]
        h = np.column_stack(
            [np.ones(visible.sum()), z[visible, t], np.ones(visible.sum())]
        )
        cross = np.einsum("naij,aj->nai", c, h)
        innovation_var = np.einsum("nai,ai->na", cross, h) + r / precision[visible, t]
        error = obs[visible, t] - np.einsum("nai,ai->na", m, h)
        means[:, visible] = m + cross * (error / innovation_var)[:, :, None]
        cov[:, visible] = (
            c
            - cross[:, :, :, None]
            * cross[:, :, None, :]
            / innovation_var[:, :, None, None]
        )
        ll[:, visible] -= 0.5 * (
            np.log(2 * np.pi * innovation_var) + error**2 / innovation_var
        )
    return means, cov, ll


def joint_future(means, cov, eta, logs):
    """Exact joint future covariance, separating conditional state and parameters."""
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.asarray(logs))
    h = np.zeros((len(eta), 5, 3))
    h[:, 0, 2] = 1
    h[:, 1:, 0], h[:, 1:, 1] = 1, PLAN
    h[:, 1:, 2] = rho[:, None] ** np.arange(1, 5)
    m = np.einsum("nki,nai->nak", h, means)
    parameter_and_state = np.einsum("nki,naij,nlj->nakl", h, cov, h)
    innovations = np.zeros((len(eta), 5, 5))
    for j in range(1, 5):
        vector = np.zeros((len(eta), 5))
        vector[:, j:] = rho[:, None] ** np.arange(5 - j)
        innovations += q * vector[:, :, None] * vector[:, None, :]
        innovations[:, j, j] += r
    # Conditional state variance given the static individual parameters.
    sol = np.linalg.solve(cov[:, :, :2, :2], cov[:, :, :2, 2, None])[:, :, :, 0]
    vs = cov[:, :, 2, 2] - np.einsum("nai,nai->na", cov[:, :, 2, :2], sol)
    loading = rho[:, None] ** np.arange(5)
    state = vs[:, :, None, None] * loading[:, None, :, None] * loading[:, None, None, :]
    if np.min(vs) < -1e-10:
        raise ValueError("negative conditional state covariance")
    return m, state + innovations[:, None], parameter_and_state - state


def score_reference(inputs, bundle, arm, evaluator, nodes=64):
    """Integrate eta by Gauss-Hermite and average unchanged source components."""
    validate_bundle(bundle)
    if arm not in ("naive_gaussian", "eligible_continuous"):
        raise ValueError("only the two Gaussian references have this analytic form")
    arrays, support = prepare(inputs, bundle, arm, evaluator)
    x, base_weights = hermgauss(nodes)
    base_weights /= np.sqrt(np.pi)
    centers, within_parts, personal_parts, effective = [], [], [], []
    for model in bundle["source_models"]:
        prior = model["prior"]
        eta = prior["mu"][2] + np.sqrt(2 * prior["Sigma"][2][2]) * x
        m, c, ll = conditional_filter(model, eta, arrays)
        logw = ll + np.log(base_weights)[:, None]
        w = np.exp(logw - logsumexp(logw, axis=0))
        means, state, ab_cov = joint_future(m, c, eta, model["log_sigma"])
        center = np.einsum("na,nak->ak", w, means)
        within = np.einsum("na,nakl->akl", w, state)
        diff = means - center[None]
        personal = np.einsum("na,nakl->akl", w, ab_cov) + np.einsum(
            "na,nak,nal->akl", w, diff, diff
        )
        centers.append(center)
        within_parts.append(within)
        personal_parts.append(personal)
        effective.append(1 / (w * w).sum(0))
    mean = np.mean(centers, axis=0)
    delta = np.asarray(centers) - mean[None]
    components = {
        "conditional_state_future": np.mean(within_parts, axis=0),
        "personal_parameter": np.mean(personal_parts, axis=0),
        "source_reader": np.einsum("bak,bal->akl", delta, delta) / len(centers),
    }
    covariance = sum(components.values())
    results = []
    for i, raw in enumerate(inputs):
        targets = {}
        for j, target in enumerate(("T2", "T3")):
            value = float(TARGETS[j] @ mean[i])
            variance = float(TARGETS[j] @ covariance[i] @ TARGETS[j])
            ref = bundle["reference"][target]
            targets[target] = {
                "score": value,
                "variance": variance,
                "lower": value - 1.96 * np.sqrt(variance),
                "upper": value + 1.96 * np.sqrt(variance),
                "reference_score": (value - ref["mu"]) / ref["sigma"],
                "components": {
                    k: float(TARGETS[j] @ c[i] @ TARGETS[j])
                    for k, c in components.items()
                },
            }
        results.append(
            {
                "arm": arm,
                "target_time": raw["target_time"],
                "future_conditions": PLAN,
                "targets": targets,
                "joint_mean": mean[i].tolist(),
                "joint_covariance": covariance[i].tolist(),
                "joint_components": {k: c[i].tolist() for k, c in components.items()},
                "eta_quadrature_nodes": nodes,
                "quadrature_effective_nodes_min": float(np.min(effective, axis=0)[i]),
                "approximation": "exact conditional augmented Gaussian filter;1D eta GH;cut-source mixture",
                **support[i],
            }
        )
    return results
