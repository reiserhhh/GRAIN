#!/usr/bin/env python3
"""Exact multi-target evidence accounts; no native-response fitting."""

import numpy as np
from scipy.special import logsumexp

TASKS = (
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
)
ARMS = ("full", "remove_selected", "remove_control", "selected_only", "control_only")


def finite(values, ndim=None):
    """Reject nonfinite or incorrectly shaped numeric observations."""
    a = np.asarray(values, float)
    if (ndim is not None and a.ndim != ndim) or not np.isfinite(a).all():
        raise ValueError("Finite observations of the declared shape required")
    return a


def digit_units(logits, normalizer):
    """Keep score, high/low log odds, geometric logit contrast and vocabulary mass."""
    a, z = finite(logits), finite(normalizer)
    if a.shape[-1] != 10 or z.shape != a.shape[:-1]:
        raise ValueError("Ten logits and same-frame vocabulary normalizers required")
    local = logsumexp(a, axis=-1)
    p = np.exp(a - local[..., None])
    return np.stack(
        (
            p @ (np.arange(10) / 9),
            logsumexp(a[..., 5:], axis=-1) - logsumexp(a[..., :5], axis=-1),
            a[..., 5:].mean(-1) - a[..., :5].mean(-1),
            np.exp(local - z),
        ),
        axis=-1,
    )


def natural_actions(values):
    """Keep deletion and isolated-subset actions in their original orientation."""
    a = finite(values)
    if a.shape[-2] != 5:
        raise ValueError("Five complete natural arms required")
    return np.stack((a[..., 1, :] - a[..., 2, :], a[..., 3, :] - a[..., 4, :]), -2)


def target_account(actions):
    """Decompose the declared target axis, retaining common and specific terms."""
    a = finite(actions)
    if a.shape[-2] != 5:
        raise ValueError("Complete five-target axis required")
    common = np.broadcast_to(a.mean(-2, keepdims=True), a.shape).copy()
    return common, a - common


def covariances(source, response):
    """Equal-author cross-covariance; views are not independent participants."""
    x, y = finite(source, 2), finite(response, 2)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Matched author families required")
    return (x - x.mean(0)).T @ (y - y.mean(0)) / (len(x) - 1)


def correlations(source, response):
    """Return undefined zero-variance contacts as NaN, never successful zero."""
    x, y = finite(source, 2), finite(response, 2)
    cov = covariances(x, y)
    den = np.outer(x.std(0, ddof=1), y.std(0, ddof=1))
    out = np.full_like(cov, np.nan)
    np.divide(cov, den, out=out, where=den > 0)
    return out


def bootstrap_diagonal(source, response, draws, seed):
    """Resample author clusters for covariance, conditional on frozen fields."""
    x, y = finite(source, 2), finite(response, 2)
    if x.shape != y.shape or x.shape[1] != 5 or draws < 1:
        raise ValueError("Five matched target columns and positive draws required")
    ix = np.random.default_rng(seed).integers(len(x), size=(draws, len(x)))
    values = np.stack([np.diag(covariances(x[i], y[i])) for i in ix])
    return np.quantile(values, [0.025, 0.975], axis=0).T


def vector_actions(vectors, selected, control):
    """Actual mean-vector deletion difference; preserve the isolated -2 identity."""
    x = finite(vectors, 2)
    s, c = np.asarray(selected), np.asarray(control)
    if (
        len(x) != 12
        or s.shape != (4,)
        or c.shape != (4,)
        or not np.issubdtype(s.dtype, np.integer)
        or not np.issubdtype(c.dtype, np.integer)
        or len(set(s) | set(c)) != 8
        or min(s.min(), c.min()) < 0
        or max(s.max(), c.max()) >= 12
    ):
        raise ValueError("Twelve events with disjoint four-event arms required")
    d = (x[c].sum(0) - x[s].sum(0)) / 8
    return np.stack((d, -2 * d))
