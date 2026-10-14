#!/usr/bin/env python3
"""Dynamic T2/T3 scoring from an arbitrary observed subset of a 48-step prefix.

Exact joint-Gaussian conditioning of (y_observed, s_47, mean future y) given each
of 64 Gauss-Hermite eta nodes, per frozen source model, with (a, b) and the AR(1)
state marginalized; node and source mixtures by total covariance. Missing steps
are simply absent rows: no zero is imputed and the clock is not compressed.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logsumexp

PLAN = np.array([1.0, -1.0, 1.0, -1.0])
HORIZON = 48


def mask_probability(kind, z, y):
    """Declared missingness mechanisms (probability an event is missing)."""
    if kind == "complete":
        return np.zeros_like(y)
    if kind == "mcar_20":
        return np.full_like(y, 0.2)
    if kind == "condition_20_60":
        return np.where(z < 0, 0.2, 0.6)
    if kind == "mnar_y":
        return 0.2 + 0.4 * expit(y)
    raise ValueError("unknown missingness mechanism")


def score_author(models, z, y, observed, nodes=64):
    """Mixture mean/cov of (T2, T3) for one author over all source models."""
    z_full = np.r_[z, PLAN]
    design = np.column_stack([np.ones(52), z_full])
    obs = np.flatnonzero(observed)
    g = np.zeros(52)
    g[48:] = 0.25
    hx, hw = hermgauss(nodes)
    lag = np.abs(np.subtract.outer(np.arange(52), np.arange(52)))
    means, covs = [], []
    for model in models:
        mu = np.asarray(model["prior"]["mu"], float)
        sig = np.asarray(model["prior"]["Sigma"], float)
        q, r = np.exp(2 * np.asarray(model["log_sigma"], float))
        eta = mu[2] + np.sqrt(2 * sig[2, 2]) * hx
        rho = 0.99 * np.tanh(eta)
        cab = sig[:2, :2] - np.outer(sig[:2, 2], sig[2, :2]) / sig[2, 2]
        mab = mu[:2][None] + (eta - mu[2])[:, None] * (sig[:2, 2] / sig[2, 2])[None]
        state = (q / (1 - rho**2))[:, None, None] * rho[:, None, None] ** lag[None]
        base = design @ cab @ design.T
        cy = base[None] + state
        c_obs = cy[:, obs][:, :, obs] + r * np.eye(len(obs))[None]
        cross = np.stack(
            [state[:, 47, obs], np.einsum("t,ktu->ku", g, cy[:, :, obs])], axis=1
        )
        prior = np.empty((nodes, 2, 2))
        prior[:, 0, 0] = state[:, 47, 47]
        prior[:, 0, 1] = prior[:, 1, 0] = np.einsum("t,kt->k", g, state[:, :, 47])
        prior[:, 1, 1] = np.einsum("t,ktu,u->k", g, cy, g) + r / 4
        mean_obs = mab @ design[obs].T
        mean_t = np.stack([np.zeros(nodes), mab @ (g @ design)], axis=1)
        if len(obs):
            chol = np.linalg.cholesky(c_obs)
            resid = (y[obs][None] - mean_obs)[:, :, None]
            white = np.linalg.solve(chol, resid)[:, :, 0]
            solved = np.linalg.solve(np.transpose(chol, (0, 2, 1)), white[:, :, None])[
                :, :, 0
            ]
            k_cross = np.linalg.solve(chol, np.transpose(cross, (0, 2, 1)))
            m_k = mean_t + np.einsum("kiu,ku->ki", cross, solved)
            v_k = prior - np.einsum("kui,kuj->kij", k_cross, k_cross)
            ll = -0.5 * np.sum(white**2, axis=1) - np.log(
                np.diagonal(chol, axis1=1, axis2=2)
            ).sum(1)
        else:
            m_k, v_k, ll = mean_t, prior, np.zeros(nodes)
        logw = ll + np.log(hw / np.sqrt(np.pi))
        w = np.exp(logw - logsumexp(logw))
        m = w @ m_k
        d = m_k - m
        v = np.einsum("k,kij->ij", w, v_k) + np.einsum("k,ki,kj->ij", w, d, d)
        means.append(m)
        covs.append(v)
    means, covs = np.array(means), np.array(covs)
    total_mean = means.mean(0)
    dm = means - total_mean
    within = covs.mean(0)
    source = np.einsum("si,sj->ij", dm, dm) / len(means)
    return total_mean, within + source, within, source
