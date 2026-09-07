#!/usr/bin/env python3
"""Finite RMS-normalization/averaging order and its quadratic mean-shift term."""

import numpy as np


def rms_normalize(x, gain, epsilon):
    """Apply the regularized, tokenwise RMSNorm map without fitting parameters."""
    x, gain = np.asarray(x, float), np.asarray(gain, float)
    if (
        x.ndim not in (1, 2)
        or not x.shape[-1]
        or gain.shape != (x.shape[-1],)
        or not np.isfinite(x).all()
        or not np.isfinite(gain).all()
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError(
            "Finite token vectors, matching gains and positive epsilon required"
        )
    return gain * x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + epsilon)


def group_projection(x, groups):
    """Replace disjoint token groups by their means; leave other positions unchanged."""
    x = np.asarray(x, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite token matrix required")
    result, used = x.copy(), set()
    for group in groups:
        ix = np.asarray(group)
        if (
            ix.ndim != 1
            or not len(ix)
            or not np.issubdtype(ix.dtype, np.integer)
            or ix.min() < 0
            or ix.max() >= len(x)
            or len(set(ix)) != len(ix)
            or used.intersection(ix)
        ):
            raise ValueError("Nonempty disjoint valid token groups required")
        result[ix] = x[ix].mean(axis=0)
        used.update(ix)
    return result


def normalization_order(x, groups, gain, epsilon):
    """Compute R(PH)-P(RH), a group-constant mean shift, not unstructured noise."""
    return rms_normalize(group_projection(x, groups), gain, epsilon) - group_projection(
        rms_normalize(x, gain, epsilon), groups
    )


def quadratic_order(mean, covariance, gain, epsilon):
    """Return minus one-half the RMSNorm Hessian contracted with within-group covariance."""
    mu, sigma, gain = (np.asarray(a, float) for a in (mean, covariance, gain))
    rms_normalize(mu, gain, epsilon)
    d = len(mu)
    if (
        mu.ndim != 1
        or sigma.shape != (d, d)
        or not np.isfinite(sigma).all()
        or not np.allclose(sigma, sigma.T)
    ):
        raise ValueError("Finite mean and symmetric matching covariance required")
    s = (mu @ mu / d + epsilon) ** -0.5
    return gain * (
        s**3 / (2 * d) * (2 * sigma @ mu + np.trace(sigma) * mu)
        - 3 * s**5 / (2 * d**2) * (mu @ sigma @ mu) * mu
    )
