#!/usr/bin/env python3
"""Sampling error of window means; event counts are precision, not population weights."""

import math

import numpy as np
from scipy.linalg import solve_triangular


def counts(n, shape):
    n = np.asarray(n, float)
    if n.shape != shape or not np.isfinite(n).all() or (n <= 0).any():
        raise ValueError("one positive effective count per observed cell required")
    return n


def eb_fit(x, n, event_var):
    """Independent units, diagonal classical error, variance in squared raw units."""
    x = np.asarray(x, float)
    n = counts(n, x.shape[:-1])
    v = np.asarray(event_var, float)
    flat = x.reshape(-1, x.shape[-1])
    if (
        len(flat) < 2
        or v.shape != (flat.shape[1],)
        or not np.isfinite(v).all()
        or (v < 0).any()
        or not np.isfinite(flat).all()
    ):
        raise ValueError("invalid source means or per-event variance")
    return {
        "mean": flat.mean(0).tolist(),
        "event_var": v.tolist(),
        "signal_var": np.maximum(0, flat.var(0, ddof=1) - v * np.mean(1 / n)).tolist(),
    }


def eb_apply(x, n, ref):
    x = np.asarray(x, float)
    n = counts(n, x.shape[:-1])
    m, s, v = (np.asarray(ref[k]) for k in ("mean", "signal_var", "event_var"))
    if x.shape[-1] != len(m) or not np.isfinite(x).all():
        raise ValueError("invalid observations")
    den = s + v / n[..., None]
    w = np.divide(s, den, out=np.ones_like(den), where=den > 0)
    return m + w * (x - m)


def split_event_variance(a, b, n_a, n_b):
    """Disjoint random half means around the SAME latent cell, not earlier/later halves."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or not len(a) or not np.isfinite(a - b).all():
        raise ValueError("paired half means required")
    v = 1 / counts(n_a, (len(a),)) + 1 / counts(n_b, (len(a),))
    return np.mean((a - b) ** 2 / v[:, None], axis=0)


def lag_rates(a, b, se1, se2, *, cross_error_cov=None):
    """EIV ratios for aligned (replicate, time, axis) trajectories; no clipped estimates."""
    a, b, s, t = (np.asarray(x, float) for x in (a, b, se1, se2))
    if (
        a.ndim != 3
        or a.shape[1] < 2
        or a.shape[0] * (a.shape[1] - 1) < 2
        or any(x.shape != a.shape or not np.isfinite(x).all() for x in (a, b, s, t))
        or min(s.min(), t.min()) < 0
    ):
        raise ValueError("invalid paired means and independent cell standard errors")

    def flat(x):
        z = x.reshape(-1, a.shape[-1])
        return z - z.mean(0)

    x, y, u, v = map(flat, (a[:, :-1], b[:, :-1], a[:, 1:], b[:, 1:]))
    cov = lambda l, r: np.mean(l * r, axis=0)
    c, lag = cov(x, y), (cov(u, y) + cov(v, x)) / 2
    n = len(x)
    if cross_error_cov is not None:
        cross = np.asarray(cross_error_cov, float)
        if (
            cross.shape != a.shape
            or not np.isfinite(cross).all()
            or (np.abs(cross) > s * t + 1e-12).any()
        ):
            raise ValueError("invalid same-cell cross-view error covariance")
        c -= (1 - 1 / n) * cross[:, :-1].mean((0, 1))
        lag += cross[:, 1:-1].sum((0, 1)) / n**2

    def private(prev, nxt, se):
        noise = se[:, :-1] ** 2
        # Centering overlapping lag samples subtracts shared interior noise.
        lag_bias = np.sum(se[:, 1:-1] ** 2, axis=(0, 1)) / n**2
        return (
            cov(prev, prev) - c - (1 - 1 / n) * noise.mean((0, 1)),
            cov(nxt, prev) - lag + lag_bias,
        )

    den1, num1 = private(x, u, s)
    den2, num2 = private(y, v, t)
    den = np.stack([c, den1, den2])
    num = np.stack([lag, num1, num2])
    return np.divide(num, den, out=np.full_like(den, np.nan), where=den > 1e-10)


def transition_scale(se, rate, structural_sd):
    se = np.asarray(se, float)
    if (
        se.ndim < 1
        or not np.isfinite(se).all()
        or (se < 0).any()
        or not np.isfinite([rate, structural_sd]).all()
        or structural_sd < 0
    ):
        raise ValueError("invalid independent cell error scales")
    return np.sqrt(structural_sd**2 + se[..., 1:] ** 2 + (rate * se[..., :-1]) ** 2)


def path_quantile(residual, scale, coverage=0.9):
    """Split conformal maximum per independent calibration trajectory, not per row."""
    r, s = np.asarray(residual, float), np.asarray(scale, float)
    if (
        s.shape != r.shape
        or r.ndim < 2
        or not r.size
        or (s <= 0).any()
        or not np.isfinite([r, s]).all()
        or not 0 < coverage < 1
    ):
        raise ValueError("finite standardized calibration paths required")
    k = math.ceil((len(r) + 1) * coverage)
    if k > len(r):
        raise ValueError("too few calibration paths for finite target quantile")
    scores = np.max(np.abs(r / s).reshape(len(r), -1), axis=1)
    return float(np.partition(scores, k - 1)[k - 1])


def whiten_system(A, b, error):
    """GLS; row SD for independent residuals or full positive definite covariance."""
    err = np.asarray(error, float)
    if not np.isfinite(err).all():
        raise ValueError("nonfinite contrast error")
    if err.shape == b.shape and np.all(err > 0):
        return A / err[:, None], b / err
    if err.shape == (len(b), len(b)) and np.allclose(err, err.T):
        L = np.linalg.cholesky(err)
        return solve_triangular(L, A, lower=True), solve_triangular(L, b, lower=True)
    raise ValueError(
        "expected positive row SD or symmetric positive definite covariance"
    )
