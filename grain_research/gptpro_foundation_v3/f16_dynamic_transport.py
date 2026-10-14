#!/usr/bin/env python3
"""Compiled conditional Gaussian targets with source and pilot uncertainty."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

PLAN = np.array([1.0, -1.0, 1.0, -1.0])


def simulate(seed, role, size):
    """Generate independent shifted-population dynamics; separate observation/truth."""
    if role not in ("pilot", "application"):
        raise ValueError("new simulation accepts only pilot/application roles")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 163, 5, 0 if role == "pilot" else 1])
    )
    a = rng.normal(1, np.sqrt(1.4), size)
    b = rng.normal(0.2, np.sqrt(0.7), size)
    rho = rng.uniform(0.2, 0.8, size)
    z = np.r_[np.where((np.arange(48) // 4) % 2 == 0, -1.0, 1.0), PLAN]
    states = np.empty((size, 52))
    current = rng.normal(size=size) * 0.5 / np.sqrt(1 - rho * rho)
    innovations = rng.normal(scale=0.5, size=(size, 52))
    noise = rng.normal(scale=0.5, size=(size, 52))
    for t in range(52):
        current = rho * current + innovations[:, t]
        states[:, t] = current
    y = a[:, None] + b[:, None] * z + states + noise
    observations = {"old": y[:, :48], "conditions": z[:48]}
    truth = {
        "a": a,
        "b": b,
        "rho": rho,
        "states": states,
        "T2": states[:, 47],
        "T3": y[:, 48:].mean(1),
        "future": y[:, 48:],
        "future_conditions": PLAN,
    }
    return observations, truth


def compile_source(model, z, nodes=64):
    """Compile stationary single-state target law, marginalizing a,b at fixed eta."""
    z = np.asarray(z, float)
    if z.ndim != 1 or len(z) != 48 or not np.isin(z, [-1, 1]).all():
        raise ValueError("declared48-event +/-1 schedule required")
    mu = np.asarray(model["prior"]["mu"])
    sigma = np.asarray(model["prior"]["Sigma"])
    x, w = hermgauss(nodes)
    eta = mu[2] + np.sqrt(2 * sigma[2, 2]) * x
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.asarray(model["log_sigma"]))
    ab_mu = mu[:2] + (eta - mu[2])[:, None] * sigma[:2, 2] / sigma[2, 2]
    ab_cov = sigma[:2, :2] - np.outer(sigma[:2, 2], sigma[2, :2]) / sigma[2, 2]
    design = np.column_stack([np.ones(48), z])
    distance = abs(np.arange(48)[:, None] - np.arange(48)[None, :])
    stationary = q / (1 - rho * rho)
    latent = stationary[:, None, None] * rho[:, None, None] ** distance
    covariance = latent + design @ ab_cov @ design.T + r * np.eye(48)
    chol = np.linalg.cholesky(covariance)
    inv = np.linalg.solve(covariance, np.broadcast_to(np.eye(48), covariance.shape))
    cross = np.zeros((nodes, 3, 48))
    cross[:, :2] = ab_cov @ design.T
    cross[:, 2] = latent[:, -1]
    prior = np.zeros((nodes, 3, 3))
    prior[:, :2, :2] = ab_cov
    prior[:, 2, 2] = stationary
    posterior = prior - cross @ inv @ cross.transpose(0, 2, 1)
    h = np.zeros((nodes, 2, 3))
    h[:, 0, 2] = 1
    h[:, 1, 0] = 1
    h[:, 1, 1] = PLAN.mean()
    loading = np.mean(rho[:, None] ** np.arange(1, 5), axis=1)
    h[:, 1, 2] = loading
    target_weight = h @ cross @ inv
    observed_mu = ab_mu @ design.T
    target_mu = np.zeros((nodes, 2))
    target_mu[:, 1] = ab_mu[:, 0] + PLAN.mean() * ab_mu[:, 1]
    offset = target_mu - np.einsum("klt,kt->kl", target_weight, observed_mu)
    target_cov = h @ posterior @ h.transpose(0, 2, 1)
    state_given_ab = posterior[:, 2, 2] - np.einsum(
        "ki,ki->k",
        posterior[:, 2, :2],
        np.linalg.solve(posterior[:, :2, :2], posterior[:, :2, 2, None])[:, :, 0],
    )
    if state_given_ab.min() < -1e-10:
        raise ValueError("negative conditional state covariance")
    load = np.column_stack([np.ones(nodes), loading])
    state_part = state_given_ab[:, None, None] * load[:, :, None] * load[:, None, :]
    personal_part = target_cov - state_part
    future_noise = np.full(nodes, r / 4)
    for j in range(1, 5):
        future_noise += q * (np.sum(rho[:, None] ** np.arange(5 - j), axis=1) / 4) ** 2
    state_part[:, 1, 1] += future_noise
    inv_mu = np.einsum("ktv,kv->kt", inv, observed_mu)
    constant = -0.5 * (
        np.einsum("kt,kt->k", observed_mu, inv_mu)
        + 2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(1)
        + 48 * np.log(2 * np.pi)
    )
    return {
        "eta": eta,
        "inverse": inv,
        "inv_mu": inv_mu,
        "constant": constant,
        "log_weights": np.log(w / np.sqrt(np.pi)),
        "target_weight": target_weight,
        "target_offset": offset,
        "state_future": state_part,
        "personal_ab": personal_part,
    }


def inverse_basis(new, z, calibrators):
    """Express every supported calibrated trajectory exactly in a common basis."""
    new, z = np.asarray(new), np.asarray(z)
    if any(c["status"] != "READY" for c in calibrators):
        return None, None, np.zeros(len(new), bool)
    mode = calibrators[0]["mode"]
    if any(c["mode"] != mode for c in calibrators):
        raise ValueError("mixed calibration modes")
    support = np.ones(len(new), bool)
    coefs = []
    if mode == "cubic":
        basis = np.stack([np.ones_like(new), new, new**3], -1)
        for c in calibrators:
            low, high = c["new_range"]
            support &= ((new >= low) & (new <= high)).all(1)
            coefs.append(c["coefficients"])
    else:
        basis = np.stack(
            [new, new * z, np.ones_like(new), np.broadcast_to(z, new.shape)], -1
        )
        for c in calibrators:
            cf = c["coefficients"]
            d0, d1, c0, c1 = (cf[0], 0, cf[1], 0) if mode == "global" else cf
            if min(abs(c0 + c1), abs(c0 - c1)) <= 1e-12:
                return None, None, np.zeros(len(new), bool)
            ip, im = 1 / (c0 + c1), 1 / (c0 - c1)
            gp, gm = -(d0 + d1) * ip, -(d0 - d1) * im
            coefs.append(
                [0.5 * (ip + im), 0.5 * (ip - im), 0.5 * (gp + gm), 0.5 * (gp - gm)]
            )
    for c in calibrators:
        if not np.isin(z, c["conditions"]).all():
            support[:] = False
    return basis[support], np.asarray(coefs), support


def component(kernel, basis, coefficients):
    """Integrate eta for all pilot draws/authors through precompiled likelihoods."""
    # Shapes: eta K, author A, time T, basis D, calibration J, target L.
    ig = np.einsum("ktv,avd->katd", kernel["inverse"], basis, optimize=True)
    quad = np.einsum("atd,kath->kadh", basis, ig, optimize=True)
    linear = np.einsum("atd,kt->kad", basis, kernel["inv_mu"], optimize=True)
    ll = -0.5 * np.einsum(
        "jd,kadh,jh->kja", coefficients, quad, coefficients, optimize=True
    )
    ll += (
        np.einsum("jd,kad->kja", coefficients, linear)
        + kernel["constant"][:, None, None]
    )
    logw = ll + kernel["log_weights"][:, None, None]
    weights = np.exp(logw - logsumexp(logw, axis=0, keepdims=True))
    tg = np.einsum("klt,atd->kald", kernel["target_weight"], basis, optimize=True)
    values = (
        np.einsum("kald,jd->kjal", tg, coefficients)
        + kernel["target_offset"][:, None, None, :]
    )
    mean = np.einsum("kja,kjal->jal", weights, values)
    delta = values - mean[None]
    state = np.einsum("kja,klm->jalm", weights, kernel["state_future"])
    personal = np.einsum("kja,klm->jalm", weights, kernel["personal_ab"])
    personal += np.einsum("kja,kjal,kjam->jalm", weights, delta, delta, optimize=True)
    return mean, state, personal, (weights * weights).sum(0) ** -1


def integrate(kernels, basis, coefficients, chunk=25):
    """Full source mixture per pilot/author; no likelihood-based source reweighting."""
    if (
        len(kernels) < 1
        or coefficients.ndim != 2
        or not np.isfinite(basis).all()
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("finite nonempty inference components required")
    records = []
    for start in range(0, len(basis), chunk):
        part = basis[start : start + chunk]
        shape = (len(coefficients), len(part), 2)
        mean = np.zeros(shape)
        second = np.zeros((*shape, 2))
        state, personal = np.zeros_like(second), np.zeros_like(second)
        ess = np.full(shape[:2], np.inf)
        for kernel in kernels:
            m, v, p, e = component(kernel, part, coefficients)
            mean += m / len(kernels)
            second += m[..., :, None] * m[..., None, :] / len(kernels)
            state += v / len(kernels)
            personal += p / len(kernels)
            ess = np.minimum(ess, e)
        source = second - mean[..., :, None] * mean[..., None, :]
        if np.min(np.linalg.eigvalsh(source)) < -1e-8:
            raise ValueError("negative source covariance beyond rounding")
        records.append(
            {
                "mean": mean,
                "state_future": state,
                "personal": personal,
                "source": source,
                "min_effective_nodes": ess,
            }
        )
    return {
        key: np.concatenate([r[key] for r in records], axis=1) for key in records[0]
    }


def collapse(moments, selected):
    """Apply total covariance across pilot calibrators, preserving all four parts."""
    raw = moments["mean"][selected]
    mean = raw.mean(0)
    centered = raw - mean
    parts = {
        key: moments[key][selected].mean(0)
        for key in ("state_future", "personal", "source")
    }
    parts["pilot"] = np.einsum("jal,jam->alm", centered, centered) / len(raw)
    return {
        "mean": mean,
        "covariance": sum(parts.values()),
        **parts,
        "min_effective_nodes": moments["min_effective_nodes"][selected].min(0),
    }
