#!/usr/bin/env python3
"""Fixed probability geometries and summary-collision measurement bounds."""

from __future__ import annotations

import math

import numpy as np


def probabilities(logits):
    """Normalize finite candidate logits without clipping or probability floors."""
    a = np.asarray(logits, float)
    if a.ndim not in (1, 2) or a.shape[-1] < 2 or not np.isfinite(a).all():
        raise ValueError("Finite vectors or matrices with two candidates required")
    e = np.exp(a - np.max(a, axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    if np.any(p <= 0):
        raise ValueError("Underflow violates the declared positive-support geometry")
    return p


def weights(value, count):
    """Validate declared normalized nonnegative weights, never estimate them."""
    w = np.asarray(value, float)
    if (
        w.shape != (count,)
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or abs(math.fsum(w) - 1) > 1e-13
    ):
        raise ValueError("Normalized declared nonnegative event weights required")
    return w


def law(value):
    """Validate a strictly positive probability vector on a fixed alphabet."""
    p = np.asarray(value, float)
    if (
        p.ndim != 1
        or len(p) < 2
        or not np.isfinite(p).all()
        or np.any(p <= 0)
        or abs(math.fsum(p) - 1) > 1e-12
    ):
        raise ValueError("Strictly positive normalized probability vector required")
    return p


def pools(logits, declared_weights):
    """Return mixture and exponential pools on identical candidate coordinates."""
    a = np.asarray(logits, float)
    if a.ndim != 2 or not len(a):
        raise ValueError("Nonempty event-by-candidate matrix required")
    p = probabilities(a)
    w = weights(declared_weights, len(a))
    mixture = w @ p
    geometric = probabilities(w @ a)
    return mixture, geometric


def kl(p, q):
    """Compute forward KL for two declared positive categorical distributions."""
    p, q = law(p), law(q)
    if p.shape != q.shape:
        raise ValueError("Same alphabet required")
    return float(p @ (np.log(p) - np.log(q)))


def barycenter_errors(logits, declared_weights, reference):
    """Check both finite-support KL minimizer identities by direct expansion."""
    a, r = np.asarray(logits, float), law(reference)
    m, g = pools(a, declared_weights)
    w, p = weights(declared_weights, len(a)), probabilities(a)
    forward_gap = math.fsum(wi * (kl(pi, r) - kl(pi, m)) for wi, pi in zip(w, p))
    reverse_gap = math.fsum(wi * (kl(r, pi) - kl(g, pi)) for wi, pi in zip(w, p))
    return {
        "mixture_kl_identity_error": abs(forward_gap - kl(m, r)),
        "geometric_kl_identity_error": abs(reverse_gap - kl(r, g)),
        "mixture_objective_gap": forward_gap,
        "geometric_objective_gap": reverse_gap,
    }


def curve_derivatives(x, y, t, values):
    """Compute exact categorical e-path score derivatives, not fitted slopes."""
    x, y, v = np.asarray(x, float), np.asarray(y, float), np.asarray(values, float)
    if x.ndim != 1 or x.shape != y.shape or x.shape != v.shape or not 0 <= t <= 1:
        raise ValueError("Aligned candidate vectors and interpolation weight required")
    if not np.isfinite(v).all():
        raise ValueError("Finite fixed answer values required")
    p = probabilities(t * x + (1 - t) * y)
    d = x - y
    score, average = float(p @ v), float(p @ d)
    centered = d - average
    return {
        "score": score,
        "first": float(p @ ((v - score) * centered)),
        "second": float(p @ ((v - score) * centered**2)),
    }


def decomposition(native, logits, declared_weights, values):
    """Separate two fixed references, retaining nonorthogonal signed cross energy."""
    n, v = law(native), np.asarray(values, float)
    m, g = pools(logits, declared_weights)
    if v.shape != n.shape or g.shape != n.shape or not np.isfinite(v).all():
        raise ValueError("Identical finite candidate values required")
    ns, bs, gs = float(n @ v), float(m @ v), float(g @ v)
    pooling, residual, total = gs - bs, ns - gs, ns - bs
    dp, dr, dt = g - m, n - g, n - m
    result = {
        "native": ns,
        "mixture": bs,
        "geometric": gs,
        "pooling_change": pooling,
        "fixed_reference_residual": residual,
        "total_change": total,
        "score_cross": 2 * pooling * residual,
        "score_identity_error": abs(total - pooling - residual),
        "score_energy_error": abs(
            total**2 - pooling**2 - residual**2 - 2 * pooling * residual
        ),
        "native_mixture_tv": float(np.abs(dt).sum() / 2),
        "native_geometric_tv": float(np.abs(dr).sum() / 2),
        "pooling_tv": float(np.abs(dp).sum() / 2),
        "probability_cross": float(2 * dp @ dr),
        "probability_energy_error": abs(
            float(dt @ dt - dp @ dp - dr @ dr - 2 * dp @ dr)
        ),
        "native_probability": n,
        "mixture_probability": m,
        "geometric_probability": g,
        **barycenter_errors(logits, declared_weights, n),
    }
    return result


def collision(native_hi, native_lo, summary_hi, summary_lo, values, tolerance):
    """Bound every deterministic readout of an identical full pooled summary."""
    hi, lo, a, b = [
        np.asarray(x, float) for x in (native_hi, native_lo, summary_hi, summary_lo)
    ]
    v = np.asarray(values, float)
    if (
        hi.ndim != 2
        or hi.shape != lo.shape
        or hi.shape != a.shape
        or hi.shape != b.shape
        or hi.shape[1:] != v.shape
        or not all(np.isfinite(x).all() for x in (hi, lo, a, b, v))
        or tolerance < 0
    ):
        raise ValueError("Aligned query-by-candidate laws required")
    maximum = float(np.max(np.abs(a - b)))
    same = maximum <= tolerance
    delta = (hi - lo) @ v
    return {
        "summary_max_difference": maximum,
        "same_summary": same,
        "native_profile_change_norm": float(np.linalg.norm(delta)),
        "native_probability_tv_mean": float(np.mean(np.abs(hi - lo).sum(axis=1) / 2)),
        "minimum_common_profile_rms": float(np.linalg.norm(delta) / 2)
        if same
        else None,
        "minimum_common_coordinate_rms": float(np.sqrt(np.mean(delta**2)) / 2)
        if same
        else None,
        "minimum_common_probability_coordinate_rms": float(
            np.sqrt(np.mean((hi - lo) ** 2)) / 2
        )
        if same
        else None,
    }
