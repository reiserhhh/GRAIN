"""HM15: fixed ordered-vector observation and a source-supported action kernel."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_ordered_moments import (
    PROPERTY,
    SPANS,
    family_weights,
    ordered_moments,
)

MODELS = ("zero_action", "ordered_rank2", "ordered_full")


def full_observation(bodies):
    """Keep six means and property moments1-3 without truncating hidden coordinates."""
    if set(bodies) != set(SPANS):
        raise ValueError("complete six-span body required; no query features")
    moments = {name: ordered_moments(bodies[name], 3) for name in SPANS}
    dimensions = {v.shape[1] for v in moments.values()}
    if len(dimensions) != 1:
        raise ValueError("all spans need the same hidden dimension")
    rows = [moments[name][0] for name in SPANS]
    rows.extend(moments[name][k] for k in (1, 2, 3) for name in PROPERTY)
    return np.asarray(rows).reshape(-1)


def projected_observation(full, frame):
    """Project each of the twelve typed moment rows through the original rank2 frame."""
    x, u = np.asarray(full, float), np.asarray(frame, float)
    if (
        x.ndim not in (1, 2)
        or u.ndim != 2
        or u.shape[1] != 2
        or x.shape[-1] != 12 * len(u)
        or not np.isfinite(x).all()
        or not np.isfinite(u).all()
        or not np.allclose(u.T @ u, np.eye(2), atol=1e-10)
    ):
        raise ValueError("complete typed moments and orthonormal rank2 frame required")
    return (x.reshape(*x.shape[:-1], 12, len(u)) @ u).reshape(*x.shape[:-1], 24)


def fit_action(source_delta, source_target, families, alpha=1.0):
    """Fit weighted dual Ridge using one source RMS norm, preserving isometric geometry."""
    x, y = np.asarray(source_delta, float), np.asarray(source_target, float)
    if (
        x.ndim != 2
        or len(x) < 2
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("finite matched source rows and positive fixed alpha required")
    w = family_weights(families)
    if len(w) != len(x):
        raise ValueError("source family count mismatch")
    scale_squared = float(w @ np.sum(x * x, axis=1) / w.sum())
    scale = float(np.sqrt(scale_squared)) if scale_squared > 0 else 1.0
    z = x / scale
    gram = z @ z.T
    gram = (gram + gram.T) / 2
    eigenvalues = np.linalg.eigvalsh(gram)
    tolerance = 100 * np.finfo(float).eps * max(1.0, float(eigenvalues[-1])) * len(x)
    if eigenvalues[0] < -tolerance:
        raise ValueError("source Gram not positive semidefinite within roundoff")
    root = np.sqrt(w)
    system = root[:, None] * gram * root[None, :] + alpha * np.eye(len(x))
    rhs = root * y
    dual = np.linalg.solve(system, rhs)
    weighted_dual = root * dual
    coefficient = x.T @ weighted_dual / scale**2
    return {
        "scale": scale,
        "scale_squared": scale_squared,
        "alpha": float(alpha),
        "intercept": 0.0,
        "centering": False,
        "source_rows": len(x),
        "source_families": len(np.unique(families)),
        "feature_dimension": x.shape[1],
        "numerical_source_rank": int(np.sum(eigenvalues > tolerance)),
        "gram_eigen_min": float(eigenvalues[0]),
        "gram_eigen_max": float(eigenvalues[-1]),
        "gram_rank_tolerance": tolerance,
        "system_residual_max": float(np.max(np.abs(system @ dual - rhs))),
        "weighted_dual": weighted_dual,
        "coefficient": coefficient,
        "eigenvalues": eigenvalues,
    }


def apply_action(delta, coefficient):
    """Score frozen action coefficients; no held-batch scales or intercept are introduced."""
    x, b = np.asarray(delta, float), np.asarray(coefficient, float)
    if (
        x.ndim != 2
        or b.shape != (x.shape[1],)
        or not np.isfinite(x).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("finite matching fixed action coordinates required")
    return x @ b


def kernel_prediction(delta, source_delta, weighted_dual, scale):
    """Evaluate the equivalent source-sample expansion without a hidden-space inverse."""
    x, s, a = (np.asarray(v, float) for v in (delta, source_delta, weighted_dual))
    if (
        x.ndim != 2
        or s.ndim != 2
        or x.shape[1] != s.shape[1]
        or a.shape != (len(s),)
        or not np.isfinite([scale]).all()
        or scale <= 0
        or not all(np.isfinite(v).all() for v in (x, s, a))
    ):
        raise ValueError(
            "finite fixed source expansion and positive source scale required"
        )
    return ((x / scale) @ (s / scale).T) @ a
