"""HM23: complete output transport with a fixed scalar input kernel."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_action_observability import moments
from grain_research.construction.math_native_joint import probability


def centered_displacement(reference, target):
    """Remove only the common logit shift, not a textual condition mean."""
    r, t = np.asarray(reference, float), np.asarray(target, float)
    if r.ndim != 2 or r.shape != t.shape or r.shape[1] < 2:
        raise ValueError("matched candidate-logit matrices required")
    if not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("finite candidate logits required")
    d = t - r
    return d - d.mean(axis=1, keepdims=True)


def source_targets(rows, reference, target, queries, states):
    """Build a source-only state-by-query matrix with complete fixed addresses."""
    if any(r["role"] != "source" for r in rows):
        raise ValueError("held address in source targets")
    if len(set(queries)) != len(queries) or len(set(states)) != len(states):
        raise ValueError("unique source states and queries required")
    layout = {(r["state_index"], r["query"]): i for i, r in enumerate(rows)}
    if len(layout) != len(rows) or set(layout) != {
        (s, q) for s in states for q in queries
    }:
        raise ValueError("complete identical query source support required")
    d = centered_displacement(reference, target)
    if len(d) != len(rows):
        raise ValueError("source address/logit size mismatch")
    return np.concatenate([d[[layout[s, q] for s in states]] for q in queries], axis=1)


def fit_multi(kernel, target, weights, unit, alpha=1.0):
    """Solve one weighted SPD system with every output column retained."""
    k, d, w = (np.asarray(v, float) for v in (kernel, target, weights))
    if (
        d.ndim != 2
        or len(d) < 2
        or not d.shape[1]
        or k.shape != (len(d), len(d))
        or w.shape != (len(d),)
        or not all(np.isfinite(v).all() for v in (k, d, w))
        or np.min(w) <= 0
        or not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-12)
        or not np.isfinite(unit)
        or unit <= 0
        or not np.isfinite(alpha)
        or alpha <= 0
        or not np.allclose(k, k.T, atol=1e-10, rtol=1e-12)
    ):
        raise ValueError("finite matched source kernel/outputs/weights required")
    gram = (k + k.T) / (2 * unit)
    eig = np.linalg.eigvalsh(gram)
    tolerance = 100 * np.finfo(float).eps * max(1.0, eig[-1]) * len(d)
    if eig[0] < -tolerance:
        raise ValueError("source kernel is not positive semidefinite")
    root = np.sqrt(w)
    system = root[:, None] * gram * root[None, :] + alpha * np.eye(len(d))
    rhs = root[:, None] * d
    dual = np.linalg.solve(system, rhs)
    return {
        "weighted_dual": root[:, None] * dual,
        "eigenvalues": eig,
        "unit": float(unit),
        "alpha": float(alpha),
        "intercept": 0.0,
        "source_rows": len(d),
        "output_columns": d.shape[1],
        "system_residual_max": float(np.max(abs(system @ dual - rhs))),
        "new_shared_systems": 1,
    }


def apply_multi(kernel, weighted_dual, unit):
    """Apply the frozen source operator without fitting application readouts."""
    k, a = np.asarray(kernel, float), np.asarray(weighted_dual, float)
    if (
        k.ndim != 2
        or a.ndim != 2
        or k.shape[1] != len(a)
        or not np.isfinite(k).all()
        or not np.isfinite(a).all()
        or not np.isfinite(unit)
        or unit <= 0
    ):
        raise ValueError("finite matched application kernel/operator required")
    return k @ a / unit


def readout(reference, delta, alphabet, anchor):
    """Preserve actual references while splitting predicted B/G/C action accounts."""
    r, d, a, bar = (np.asarray(v, float) for v in (reference, delta, alphabet, anchor))
    if (
        r.ndim != 2
        or r.shape != d.shape
        or a.shape != (r.shape[1],)
        or bar.shape != a.shape
        or not all(np.isfinite(v).all() for v in (r, d, a, bar))
    ):
        raise ValueError("finite reference/motion/alphabet/covector required")
    p = np.array([probability(z) for z in r])
    pt = np.array([probability(z) for z in r + d])
    y = (pt - p) @ a
    g = p * (a - (p @ a)[:, None])
    b = d @ bar
    tangent = np.sum(g * d, axis=1)
    return y, np.column_stack((b, tangent - b, y - tangent))


def scalar_metrics(native, predicted, weights):
    """Report weighted action error and association in the original score unit."""
    y, p = np.asarray(native, float), np.asarray(predicted, float)
    if y.ndim != 1 or y.shape != p.shape:
        raise ValueError("matched scalar actions required")
    cov = moments(np.column_stack((y, p)), weights)["covariance"]
    w = np.asarray(weights, float)
    mse = float(w @ (p - y) ** 2)
    norm = float(np.sqrt(cov[0, 0] * cov[1, 1]))
    return {
        "native_rms": float(np.sqrt(w @ y**2)),
        "predicted_rms": float(np.sqrt(w @ p**2)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(w @ abs(p - y)),
        "pearson": float(cov[0, 1] / norm) if norm > 0 else None,
    }


def remainder_covariance(native_a, native_b, predicted_a, predicted_b, weights):
    """Keep all four signed prediction/remainder contributions, including cancellation."""
    y, z, p, q = (
        np.asarray(v, float) for v in (native_a, native_b, predicted_a, predicted_b)
    )
    if y.ndim != 1 or any(v.shape != y.shape for v in (z, p, q)):
        raise ValueError("matched query action pairs required")
    c = moments(np.column_stack((p, y - p, q, z - q)), weights)["covariance"]
    terms = {"PP": c[0, 2], "PR": c[0, 3], "RP": c[1, 2], "RR": c[1, 3]}
    native = moments(np.column_stack((y, z)), weights)["covariance"][0, 1]
    return {
        **{k: float(v) for k, v in terms.items()},
        "native_covariance": float(native),
        "recombination_error": float(abs(native - sum(terms.values()))),
    }
