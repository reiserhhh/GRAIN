"""Finite observation geometry of fixed bilinear context actions (HM18)."""

from __future__ import annotations

from collections import Counter

import numpy as np

CHANNELS = ("linear", "symmetric", "skew", "finite_step")


def family_weights(families):
    """Assign equal total mass to each family and equal mass within a family."""
    counts = Counter(families)
    if not counts:
        raise ValueError("empty observation design")
    return np.asarray([1.0 / (len(counts) * counts[f]) for f in families])


def validated_weights(weights, n):
    """Validate a probability measure rather than silently renormalizing it."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("invalid observation weights")
    if not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("observation weights must sum to one")
    return w


def moments(values, weights):
    """Return weighted uncentered Gram, covariance and mean for action columns."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or not np.all(np.isfinite(v)):
        raise ValueError("finite action matrix required")
    w = validated_weights(weights, len(v))
    mean = w @ v
    gram = v.T @ (w[:, None] * v)
    centered = v - mean
    covariance = centered.T @ (w[:, None] * centered)
    return {"mean": mean, "gram": gram, "covariance": covariance}


def spectrum(gram, multiplier=100.0):
    """Use a fixed machine threshold; do not interpret rank as trait count."""
    g = np.asarray(gram, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.all(np.isfinite(g)):
        raise ValueError("finite square Gram required")
    eigen = np.linalg.eigvalsh((g + g.T) / 2)
    tolerance = multiplier * np.finfo(float).eps * max(1.0, eigen[-1]) * len(g)
    if eigen[0] < -tolerance:
        raise ValueError("observation Gram is not positive semidefinite")
    positive = eigen[eigen > tolerance]
    return {
        "eigenvalues": eigen,
        "tolerance": float(tolerance),
        "rank": len(positive),
        "smallest_positive": float(positive[0]) if len(positive) else None,
        "largest": float(eigen[-1]),
        "condition_number": float(positive[-1] / positive[0])
        if len(positive) == len(g)
        else None,
    }


def dyad_grams(x, context):
    """Contract full, symmetric and skew dyads without materializing d^2 columns."""
    x, c = np.asarray(x, dtype=float), np.asarray(context, dtype=float)
    if x.ndim != 2 or x.shape != c.shape:
        raise ValueError("paired equal-dimensional observation coordinates required")
    direct = (x @ x.T) * (c @ c.T)
    cross = x @ c.T
    transpose = cross * cross.T
    return {
        "A": direct,
        "S": (direct + transpose) / 2,
        "H": (direct - transpose) / 2,
    }


def weighted_design_gram(gram, weights):
    """Scale dyad rows by sqrt(mass), preserving the observation kernel."""
    g = np.asarray(gram, dtype=float)
    w = validated_weights(weights, len(g))
    return np.sqrt(w[:, None] * w[None, :]) * g


def cosine(inner, norm_a_squared, norm_b_squared):
    """Leave zero-action similarity undefined, including exact no-op designs."""
    if norm_a_squared <= 0 or norm_b_squared <= 0:
        return None
    return float(inner / np.sqrt(norm_a_squared * norm_b_squared))


def action_diagnostics(values, weights, multiplier=100.0):
    """Distinguish S/H near-collinearity from a numerically exact null direction."""
    v = np.asarray(values, dtype=float)
    if v.shape[1] != len(CHANNELS):
        raise ValueError("linear/S/H/step columns required")
    m = moments(v, weights)
    g, cov = m["gram"], m["covariance"]
    sh = g[1:3, 1:3]
    eig = spectrum(sh, multiplier)
    s, h = v[:, 1], v[:, 2]
    w = validated_weights(weights, len(v))
    direct_rms = float(np.sqrt(w @ (s + h) ** 2))
    transpose_rms = float(np.sqrt(w @ (s - h) ** 2))
    rms = np.sqrt(np.maximum(0.0, np.diag(g)))
    trace = float(np.trace(sh))
    return {
        **{f"{name}_rms": float(rms[i]) for i, name in enumerate(CHANNELS)},
        "sh_inner": float(g[1, 2]),
        "sh_cosine": cosine(g[1, 2], g[1, 1], g[2, 2]),
        "sh_correlation": cosine(cov[1, 2], cov[1, 1], cov[2, 2]),
        "direct_rms": direct_rms,
        "transpose_rms": transpose_rms,
        "transpose_direct_ratio": transpose_rms / direct_rms
        if direct_rms > 0
        else None,
        "step_skew_ratio": float(rms[3] / rms[2]) if rms[2] > 0 else None,
        "sh_eigen_min": float(eig["eigenvalues"][0]),
        "sh_eigen_max": float(eig["eigenvalues"][-1]),
        "sh_rank": eig["rank"],
        "sh_tolerance": eig["tolerance"],
        "sh_condition_number": eig["condition_number"],
        "sh_min_trace_ratio": float(eig["eigenvalues"][0] / trace)
        if trace > 0
        else None,
        "numerical_null_dimension": 2 - eig["rank"],
    }
