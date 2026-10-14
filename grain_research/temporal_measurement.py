#!/usr/bin/env python3
"""Joint conditional profiles and noisy continuous-time state observations.

Linear Gaussian conditioning and multivariate OU transitions are inherited
methods. The integration specifies which target and observation clock are used;
past states are smoothed, while the last state uses only available past data.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import pairwise

import numpy as np
from scipy.linalg import cho_factor, cho_solve, expm


def validate_covariance(value, *, definite=False):
    """Require an explicit symmetric positive (semi)definite covariance."""
    value = np.asarray(value, float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("Square covariance required")
    if not np.isfinite(value).all() or not np.allclose(value, value.T, atol=1e-12):
        raise ValueError("Finite symmetric covariance required")
    threshold = 1e-12 if definite else -1e-10
    if np.linalg.eigvalsh(value).min() < threshold:
        raise ValueError("Invalid covariance eigenvalue")
    return value


def time_axis(times):
    """Validate actual times or an explicitly declared order-based index."""
    times = np.asarray(times, float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("At least two finite time points required")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Strictly increasing times required; do not fabricate ties")
    return times - times[0]


@lru_cache(maxsize=8192)
def _transition(flat, q, delta):
    """Cache immutable matrix exponentials, not fitted observations or outcomes."""
    value = expm(np.asarray(flat).reshape(q, q) * delta)
    value.setflags(write=False)
    return value


def state_covariance(times, drift, stationary):
    """Exact stationary multivariate OU covariance at arbitrary recorded times."""
    times = time_axis(times)
    a, p = np.asarray(drift, float), validate_covariance(stationary, definite=True)
    q = len(p)
    if a.shape != p.shape or not np.isfinite(a).all():
        raise ValueError("Matching finite drift required")
    if np.linalg.eigvals(a).real.max() >= 0:
        raise ValueError("Stable drift required")
    validate_covariance(-(a @ p + p @ a.T))
    result = np.zeros((len(times) * q, len(times) * q))
    key = tuple(a.ravel())
    for i in range(len(times)):
        for j in range(i + 1):
            block = _transition(key, q, float(times[i] - times[j])) @ p
            result[i * q : (i + 1) * q, j * q : (j + 1) * q] = block
            result[j * q : (j + 1) * q, i * q : (i + 1) * q] = block.T
    return result


def observation_design(condition, partner):
    """Per-channel level/condition/partner columns with chronological rows."""
    z, u = np.asarray(condition, float), np.asarray(partner, float)
    if z.ndim != 2 or u.shape != (len(z),):
        raise ValueError("Condition matrix and one partner input per occasion required")
    if not np.isfinite(z).all() or not np.isfinite(u).all():
        raise ValueError("Finite observed design required")
    n, q = z.shape
    x = np.zeros((n * q, 3 * q))
    for k in range(q):
        x[k::q, 3 * k : 3 * k + 3] = np.c_[np.ones(n), z[:, k], u]
    return x


def posterior(
    times,
    evidence,
    condition,
    partner,
    *,
    drift,
    stationary,
    noise,
    profile_mean,
    profile_cov,
):
    """One joint posterior from noisy evidence, without latent truth as input.

    No automatic parameter fitting or empirical time-unit selection is performed.
    Kernel columns map the actual observed evidence to every posterior quantity.
    Missing evidence entries remain missing; their unobserved states can borrow
    from the declared model but must not be reported as direct observations.
    """
    t = time_axis(times)
    y, z = np.asarray(evidence, float), np.asarray(condition, float)
    if y.ndim != 2 or z.shape != y.shape or len(y) != len(t) or np.isinf(y).any():
        raise ValueError("Aligned finite-or-missing evidence required")
    n, q = y.shape
    mean = np.asarray(profile_mean, float)
    v = validate_covariance(profile_cov, definite=True)
    r = validate_covariance(noise, definite=True)
    if mean.shape != (3 * q,) or v.shape != (3 * q, 3 * q) or r.shape != (q, q):
        raise ValueError("Profile and measurement dimensions differ")
    if not np.isfinite(mean).all():
        raise ValueError("Finite source reference required")
    s = state_covariance(t, drift, stationary)
    x = observation_design(z, partner)
    prior = np.zeros((3 * q + n * q, 3 * q + n * q))
    prior[: 3 * q, : 3 * q], prior[3 * q :, 3 * q :] = v, s
    h = np.c_[x, np.eye(n * q)]
    observed = np.isfinite(y.ravel())
    if not observed.any():
        raise ValueError("No observed evidence")
    ho = h[observed]
    rr = np.kron(np.eye(n), r)[np.ix_(observed, observed)]
    factor = cho_factor(ho @ prior @ ho.T + rr, lower=True)
    gain = cho_solve(factor, ho @ prior).T
    prior_mean = np.r_[mean, np.zeros(n * q)]
    estimate = prior_mean + gain @ (y.ravel()[observed] - ho @ prior_mean)
    covariance = prior - gain @ ho @ prior
    covariance = (covariance + covariance.T) / 2
    kernels = np.zeros((len(estimate), n * q))
    kernels[:, observed] = gain
    return {
        "mean": estimate,
        "covariance": covariance,
        "kernels": kernels,
        "offset": prior_mean - gain @ ho @ prior_mean,
        "observed": observed,
        "design": x,
        "times": t,
        "channels": q,
    }


def target_vector(result, kind, channel, *, z=0.0, partner=0.0):
    """Define the estimand before selecting a kernel or interpreting a score."""
    q, t = result["channels"], result["times"]
    if not 0 <= channel < q:
        raise ValueError("Unknown channel")
    out = np.zeros(len(result["mean"]))
    if kind in ("level", "condition", "partner"):
        out[3 * channel + ("level", "condition", "partner").index(kind)] = 1
    elif kind in ("current_state", "current_response"):
        out[3 * q + (len(t) - 1) * q + channel] = 1
        if kind == "current_response":
            out[3 * channel : 3 * channel + 3] = [1, z, partner]
    elif kind in ("window_state", "window_response"):
        # Trapezoid of sampled states, not the exact continuous-time integral.
        gaps = np.diff(t)
        weights = np.r_[gaps[0] / 2, (gaps[:-1] + gaps[1:]) / 2, gaps[-1] / 2] / t[-1]
        out[3 * q + channel :: q] = weights
        if kind == "window_response":
            out[3 * channel : 3 * channel + 3] = [1, z, partner]
    elif kind == "conditional_profile":
        out[3 * channel : 3 * channel + 3] = [1, z, partner]
    else:
        raise ValueError("Unknown measurement target")
    return out


def score_target(result, target):
    """Score, model-conditional interval and explicit signed evidence kernel."""
    w = np.asarray(target, float)
    if w.shape != result["mean"].shape or not np.isfinite(w).all():
        raise ValueError("Aligned finite target required")
    value = float(w @ result["mean"])
    variance = float(w @ result["covariance"] @ w)
    if variance < -1e-9:
        raise ArithmeticError("Negative posterior target variance")
    radius = 1.959963984540054 * np.sqrt(max(variance, 0))
    return {
        "score": value,
        "variance": max(variance, 0),
        "lower": value - radius,
        "upper": value + radius,
        "kernel": w @ result["kernels"],
        "offset": float(w @ result["offset"]),
    }


def interaction_pairs(rows):
    """Use adjacent observed assistant/user turns only, preserving real indices."""
    required = {"role", "turn_sequence"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("Role and original turn sequence required")
    order = sorted(rows, key=lambda r: r["turn_sequence"])
    if len({r["turn_sequence"] for r in order}) != len(order):
        raise ValueError("Duplicate turn sequence")
    return [
        (left, right)
        for left, right in pairwise(order)
        if left["role"] == "assistant"
        and right["role"] == "user"
        and right["turn_sequence"] == left["turn_sequence"] + 1
    ]
