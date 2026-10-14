#!/usr/bin/env python3
"""Standard conditional-response and Gaussian design references for GRAIN.

These are known mathematical models, not claims about human latent truth.
Observed opportunities and the meaning of the target must be declared separately.
"""

from __future__ import annotations

import numpy as np


def basis(z, degree=1):
    """Return intercept and polynomial condition terms, preserving interactions."""
    z = np.asarray(z, dtype=float)
    if degree not in (1, 2, 3) or not np.isfinite(z).all():
        raise ValueError("Finite conditions and degree 1, 2 or 3 required")
    return np.stack([z**k for k in range(degree + 1)], axis=-1)


def response_coordinates(z, evidence, degree=1, penalty=0.1):
    """Estimate each author's response using only that author's observations."""
    z, evidence = np.asarray(z), np.asarray(evidence)
    if z.ndim != 2 or z.shape != evidence.shape or penalty <= 0:
        raise ValueError(
            "Aligned author-by-observation matrices and positive penalty required"
        )
    if not np.isfinite(evidence).all():
        raise ValueError("Evidence must be finite; missing evidence is not zero")
    design = basis(z, degree)
    gram = np.einsum("uti,utj->uij", design, design)
    rhs = np.einsum("uti,ut->ui", design, evidence)
    return np.linalg.solve(gram + penalty * np.eye(degree + 1), rhs[..., None])[..., 0]


def posterior_covariance(design, noise_variance, prior_covariance):
    """Gaussian linear posterior covariance with known independent error variance."""
    design = np.asarray(design, dtype=float)
    noise = np.broadcast_to(np.asarray(noise_variance, dtype=float), (len(design),))
    prior = np.asarray(prior_covariance, dtype=float)
    if prior.shape != (design.shape[1], design.shape[1]) or np.any(noise <= 0):
        raise ValueError("Incompatible covariance or nonpositive variance")
    if not np.allclose(prior, prior.T) or np.linalg.eigvalsh(prior).min() <= 0:
        raise ValueError("Prior covariance must be symmetric positive definite")
    return np.linalg.inv(np.linalg.inv(prior) + (design.T / noise) @ design)


def target_variance_gain(covariance, target, candidate, noise_variance):
    """Sherman-Morrison reduction of posterior variance for target c.T beta."""
    covariance, target, candidate = map(np.asarray, (covariance, target, candidate))
    if noise_variance <= 0:
        raise ValueError("Observation variance must be positive")
    vector = covariance @ candidate
    return float((target @ vector) ** 2 / (noise_variance + candidate @ vector))


def gaussian_scores(design, observations, noise, prior_covariance):
    """Score new authors with a fixed zero-mean prior; do not estimate it on them."""
    cov = posterior_covariance(design, noise, prior_covariance)
    score = (np.asarray(observations) / noise) @ design @ cov
    return score, cov
