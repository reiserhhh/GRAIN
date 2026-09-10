#!/usr/bin/env python3
"""Reference-centered action calibration and output-aware interaction geometry."""

from __future__ import annotations

import numpy as np


def fit_support_correction(displacements, target, prior, penalty=0.01):
    """Fit one zero-intercept dual Ridge correction about a fixed reference vector."""
    d, y, g = (np.asarray(x, float) for x in (displacements, target, prior))
    if (
        d.ndim != 2
        or len(d) < 2
        or y.shape != (len(d),)
        or g.shape != (d.shape[1],)
        or not all(np.isfinite(x).all() for x in (d, y, g))
        or not np.isfinite(penalty)
        or penalty <= 0
    ):
        raise ValueError(
            "Finite aligned reference rows/targets/prior and positive penalty required"
        )
    k = d @ d.T
    scale = float(np.trace(k) / len(d))
    if scale <= 1e-20:
        raise ValueError("Reference displacements have no energy")
    regularization = len(d) * penalty * scale
    gram = k + regularization * np.eye(len(d))
    coefficients = np.linalg.solve(gram, y - d @ g)
    correction = d.T @ coefficients
    s = np.linalg.svd(d, full_matrices=False, compute_uv=False)
    return {
        "reference_rows": d.copy(),
        "prior": g.copy(),
        "correction": correction,
        "coefficients": coefficients,
        "regularization": regularization,
        "scale": scale,
        "penalty": float(penalty),
        "singular_values": s,
        "effective_df": float(np.trace(np.linalg.solve(gram, k))),
    }


def read_support(displacements, model):
    """Read through frozen prior and support correction without normalizing new inputs."""
    d = np.asarray(displacements, float)
    if d.ndim != 2 or d.shape[1] != len(model["prior"]) or not np.isfinite(d).all():
        raise ValueError(
            "Finite displacement rows matching the reference model required"
        )
    return d @ (model["prior"] + model["correction"])


def soft_support(displacements, model):
    """Measure displacement energy covered by the reference Ridge support operator."""
    d = np.asarray(displacements, float)
    ref = np.asarray(model["reference_rows"], float)
    if d.ndim != 2 or d.shape[1] != ref.shape[1] or not np.isfinite(d).all():
        raise ValueError("Finite aligned displacement rows required")
    cross = d @ ref.T
    gram = ref @ ref.T + model["regularization"] * np.eye(len(ref))
    numerator = np.sum(cross * np.linalg.solve(gram, cross.T).T, axis=1)
    denominator = np.sum(d * d, axis=1)
    return np.divide(
        numerator, denominator, out=np.zeros(len(d)), where=denominator > 0
    )


def axis_account(actual, predicted, direction):
    """Exact parallel/perpendicular error decomposition about a pre-frozen unit axis."""
    y, p, v = (np.asarray(x, float) for x in (actual, predicted, direction))
    if (
        y.ndim != 2
        or p.shape != y.shape
        or v.shape != (y.shape[1],)
        or not all(np.isfinite(x).all() for x in (y, p, v))
        or abs(float(v @ v) - 1) > 1e-10
    ):
        raise ValueError(
            "Finite aligned vectors and a unit reference direction required"
        )
    a, b = y @ v, p @ v
    yr, pr = y - a[:, None] * v, p - b[:, None] * v
    epar = (a - b) ** 2
    eperp = np.sum((yr - pr) ** 2, axis=1)
    total = np.sum((y - p) ** 2, axis=1)
    return {
        "actual_amplitude": a,
        "predicted_amplitude": b,
        "actual_perpendicular_norm": np.linalg.norm(yr, axis=1),
        "predicted_perpendicular_norm": np.linalg.norm(pr, axis=1),
        "actual_energy": np.sum(y * y, axis=1),
        "parallel_energy": a * a,
        "parallel_error": epar,
        "perpendicular_error": eperp,
        "total_error": total,
        "closure_error": abs(total - epar - eperp),
    }


def output_gradient(logits, values):
    """Analytic gradient of the conditional alphabet score, in its native units."""
    z, v = np.asarray(logits, float), np.asarray(values, float)
    if z.ndim != 1 or z.shape != v.shape or not np.isfinite([z, v]).all():
        raise ValueError("Finite matching answer logits and numeric values required")
    p = np.exp(z - z.max())
    p /= p.sum()
    return p * (v - p @ v)
