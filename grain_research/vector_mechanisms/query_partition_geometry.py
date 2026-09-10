#!/usr/bin/env python3
"""Finite log-partition moments joining query selection and GRAIN directions."""

import numpy as np

from .fixed_seed_attention import rotary
from .source_query_selection import key_actions

FIELDS = (
    "log_mass",
    "mean_v",
    "mean_d",
    "var_v",
    "var_d",
    "uncentered",
    "centering",
    "response",
    "rho",
    "effective_tokens",
)


def partition(logits, features, theta=None):
    """Return log-mean partition, gradient and Hessian on finite masked support.

    Features are fixed sufficient statistics, not functions of theta. Negative
    infinity logits denote forbidden tokens; other nonfinite inputs are rejected.
    """
    l, x = np.asarray(logits, float), np.asarray(features, float)
    if (
        l.ndim != 1
        or x.ndim != 2
        or len(l) != len(x)
        or not len(l)
        or x.shape[1] < 1
        or np.isnan(l).any()
        or np.isposinf(l).any()
        or not np.isfinite(l).any()
        or not np.isfinite(x).all()
    ):
        raise ValueError("Finite features and nonempty allowed support required")
    t = np.zeros(x.shape[1]) if theta is None else np.asarray(theta, float)
    if t.shape != (x.shape[1],) or not np.isfinite(t).all():
        raise ValueError("Aligned finite parameter required")
    allowed = np.isfinite(l)
    x, l = x[allowed], l[allowed]
    score = l + x @ t
    if not np.isfinite(score).all():
        raise ValueError("Finite tilted logits required")
    mx = score.max()
    p = np.exp(score - mx)
    z = p.sum()
    p /= z
    mu = p @ x
    # A constant statistic has exactly zero centered response, including roundoff.
    fixed = np.ptp(x, axis=0) == 0
    mu[fixed] = x[0, fixed]
    centered = x - mu
    cov = (centered * p[:, None]).T @ centered
    return {
        "log_mass": float(mx + np.log(z) - np.log(len(l))),
        "mean": mu,
        "covariance": cov,
        "raw_moment": (x * p[:, None]).T @ x,
        "effective_tokens": float(1 / (p @ p)),
    }


def compile_geometry(query_anchors, evidence):
    """Compile [anchor,mode,factor,head] moments with original GQA/RoPE masks.

    The independent earlier key-action routine validates the original input
    contract and checks the head-summed covariance; it reads no native targets.
    """
    direct = key_actions(query_anchors, evidence)
    e = {k: np.asarray(v, float) for k, v in evidence.items()}
    q = rotary(np.asarray(query_anchors, float), e["cos"][-1], e["sin"][-1])
    k = rotary(e["k"], e["cos"][:, None], e["sin"][:, None])
    dk = rotary(e["dk"], e["cos"][None, :, None], e["sin"][None, :, None])
    groups = q.shape[1] // k.shape[1]
    shape = (len(q), 2, len(dk), q.shape[1])
    out = {name: np.zeros(shape) for name in FIELDS}
    for a in range(len(q)):
        for h in range(q.shape[1]):
            kv = h // groups
            logits = k[:, kv] @ q[a, h] * float(e["scale"]) + e["mask"]
            values = e["v"][:, kv] @ e["g"][:, h].T
            directions = np.einsum("fth,h->tf", dk[:, :, kv], q[a, h]) * float(
                e["scale"]
            )
            m = partition(logits, np.column_stack((values, directions)))
            mu, cov, raw = m["mean"], m["covariance"], m["raw_moment"]
            for mode in range(2):
                vv = cov[mode, mode]
                vd = np.diag(cov)[2:]
                r = cov[mode, 2:]
                den = np.sqrt(vv * vd)
                rho = np.divide(r, den, out=np.full(len(dk), np.nan), where=den > 0)
                vals = (
                    m["log_mass"],
                    mu[mode],
                    mu[2:],
                    vv,
                    vd,
                    raw[mode, 2:],
                    mu[mode] * mu[2:],
                    r,
                    rho,
                    m["effective_tokens"],
                )
                for name, val in zip(FIELDS, vals, strict=True):
                    out[name][a, mode, :, h] = val
    np.testing.assert_allclose(out["response"].sum(-1), direct, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(
        out["uncentered"] - out["centering"], out["response"], atol=1e-11, rtol=1e-10
    )
    return out


def summary(fields):
    """Equal-cell geometry with signed heads retained until final summation.

    Input fields are [author,factor,head] or [author,head]. Ratios pool their
    declared numerators/denominators; they are not causal variance attributions.
    """
    r, u, b = (
        np.asarray(fields[k], float) for k in ("response", "uncentered", "centering")
    )
    spread = np.sqrt(fields["var_v"] * fields["var_d"])
    norm = lambda v: float(np.sqrt(np.mean(np.square(v))))
    ratio = lambda n, d: float(n / d) if d > 0 else np.nan
    rho = np.asarray(fields["rho"])
    return {
        "mean_value": float(np.mean(fields["mean_v"])),
        "mean_direction": float(np.mean(fields["mean_d"])),
        "rms_value_sd": float(np.sqrt(np.mean(fields["var_v"]))),
        "rms_direction_sd": float(np.sqrt(np.mean(fields["var_d"]))),
        "mean_spread_product": float(np.mean(spread)),
        "spread_weighted_abs_alignment": ratio(np.abs(r).sum(), spread.sum()),
        "head_sign_retention": ratio(np.abs(r.sum(-1)).sum(), np.abs(r).sum()),
        "uncentered_sum_rms": norm(u.sum(-1)),
        "centering_sum_rms": norm(b.sum(-1)),
        "response_sum_rms": norm(r.sum(-1)),
        "centered_vs_raw_abs_ratio": ratio(
            np.abs(r.sum(-1)).sum(), np.abs(u.sum(-1)).sum()
        ),
        "mean_effective_tokens": float(np.mean(fields["effective_tokens"])),
        "undefined_rho_count": int((~np.isfinite(rho)).sum()),
        "cells": int(r.size),
        "moment_closure_max": float(np.max(np.abs(u - b - r))),
    }
