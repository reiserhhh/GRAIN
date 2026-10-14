#!/usr/bin/env python3
"""Gaussian observation pushforwards with explicit canonical evidence units."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_vector_likelihood as V

PROGRAMS = (
    "atomic",
    "pooled",
    "balanced",
    "unequal",
    "overlap",
    "aliases",
    "lossy_prefix",
)


def programs(n):
    """Return fixed groupings; aliases and overlaps refer to existing root events."""
    if not isinstance(n, int) or n < 4:
        raise ValueError("At least four canonical events required")
    eye = np.eye(n)
    mean = np.ones((1, n)) / n
    half = n // 2
    balanced = np.stack((eye[:half].mean(0), eye[half:].mean(0)))
    return {
        "atomic": eye,
        "pooled": mean,
        "balanced": balanced,
        "unequal": np.stack((eye[0], eye[1:].mean(0))),
        "overlap": np.vstack((eye[: half + 1].mean(0), eye[half - 1 :].mean(0), mean)),
        "aliases": np.vstack((eye, eye[:2], eye[:2])),
        "lossy_prefix": eye[:half].mean(0, keepdims=True),
    }


def geometry(a, sigma2, tau2=0.0, diagonal=False):
    """Push the root covariance through a grouping, without fitted corrections."""
    a = np.asarray(a, float)
    if (
        a.ndim != 2
        or min(a.shape) < 1
        or not np.isfinite(a).all()
        or np.any(a < 0)
        or not np.allclose(a.sum(1), 1, atol=1e-12, rtol=0)
        or not np.isfinite(sigma2)
        or sigma2 <= 0
        or not np.isfinite(tau2)
        or tau2 < 0
    ):
        raise ValueError(
            "Stochastic grouping and declared positive covariance required"
        )
    n = a.shape[1]
    root = sigma2 * np.eye(n) + tau2 * np.ones((n, n))
    covariance = a @ root @ a.T
    if diagonal:
        covariance = np.diag(np.diag(covariance))
    values, vectors = np.linalg.eigh(covariance)
    tol = 64 * np.finfo(float).eps * len(values) * values.max()
    if values.min() < -tol:
        raise ValueError("Invalid induced covariance")
    keep = values > tol
    inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    support = vectors[:, keep] @ vectors[:, keep].T
    c = a @ np.ones(n)
    beta = inverse @ c
    alpha = float(c @ beta)
    if alpha <= 0 or np.max(np.abs(c - support @ c)) > 1e-9:
        raise ValueError("Source mean lies outside observation support")
    root_weights = a.T @ beta / alpha
    return {
        "covariance": covariance,
        "inverse": inverse,
        "support": support,
        "beta": beta,
        "alpha": alpha,
        "root_weights": root_weights,
        "rank": int(keep.sum()),
        "root_information": n / (sigma2 + n * tau2),
        "mean_recovery_defect": float(np.linalg.norm(root_weights - 1 / n)),
    }


def posterior(prototypes, labels, centers, precisions):
    """Read all targets from the same positive source-mixture observation law."""
    z, y, _ = V.validate(prototypes, labels, 1.0)
    x, alpha = np.asarray(centers, float), np.asarray(precisions, float)
    if (
        x.ndim != 2
        or x.shape[1] != z.shape[1]
        or alpha.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(alpha).all()
        or np.any(alpha <= 0)
    ):
        raise ValueError("Addressed centers and positive information required")
    # The observation-only quadratic term cancels; source-dependent terms remain.
    log_law = alpha[:, None] * (x @ z.T - (z * z).sum(1)[None, :] / 2)
    w = np.exp(log_law - logsumexp(log_law, axis=1)[:, None])
    profile = (y @ (2 ** np.arange(5))).astype(int)
    mu = w @ y
    odds = np.stack(
        [
            logsumexp(log_law[:, y[:, q] == 1], axis=1)
            - logsumexp(log_law[:, y[:, q] == 0], axis=1)
            for q in range(5)
        ],
        axis=1,
    )
    return {
        "posterior": mu,
        "relative": mu - y.mean(0),
        "profile": w @ np.eye(32)[profile],
        "log_odds": odds,
    }


def cold_observation(record, dimensions):
    """Compile grouping weights against immutable base-event IDs, not chunk IDs."""
    if set(record) != {"event_ids", "vectors", "grouping"}:
        raise ValueError("Only canonical identified vectors and grouping allowed")
    ids = record["event_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Repeated or missing canonical event identity")
    x = np.asarray(record["vectors"], float)
    if x.shape != (len(ids), dimensions) or not np.isfinite(x).all():
        raise ValueError("Canonical vectors have changed support")
    a = np.asarray(record["grouping"], float)
    if a.ndim != 2 or a.shape[1] != len(ids):
        raise ValueError("Grouping cannot invent new root evidence")
    geometry(a, 1.0)
    return x, a @ x


def read_grouped(prototypes, labels, observed, a, sigma2, tau2=0.0, diagonal=False):
    """Score a supplied grouped observation and reject impossible singular inputs."""
    g = np.asarray(observed, float)
    info = geometry(a, sigma2, tau2, diagonal)
    if g.ndim != 2 or g.shape[0] != len(info["beta"]) or not np.isfinite(g).all():
        raise ValueError("Finite grouped vector observations required")
    if np.linalg.norm(g - info["support"] @ g) > 1e-9 * max(1, np.linalg.norm(g)):
        raise ValueError("Grouped observation is outside induced covariance support")
    center = info["beta"] @ g / info["alpha"]
    return posterior(prototypes, labels, center[None, :], np.array([info["alpha"]]))


def literal(prototypes, labels, observed, a, sigma2, tau2):
    """Alternate full quadratic likelihood, retaining the source-common residual."""
    z, y, _ = V.validate(prototypes, labels, 1.0)
    g = np.asarray(observed, float)
    info = geometry(a, sigma2, tau2)
    values = []
    for prototype in z:
        residual = g - prototype
        values.append(-float(np.sum(residual * (info["inverse"] @ residual))) / 2)
    k = np.exp(np.asarray(values) - max(values))
    return (k / k.sum()) @ y
