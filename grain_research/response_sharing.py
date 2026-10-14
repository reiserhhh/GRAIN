#!/usr/bin/env python3
"""Source-learned response covariance, separate from condition-description distance."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def project_psd(matrix):
    """Nearest symmetric PSD matrix in Frobenius norm; retain raw eigenvalues."""
    a = np.asarray(matrix, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("Finite square covariance required")
    eig, vec = np.linalg.eigh((a + a.T) / 2)
    return (vec * np.maximum(eig, 0)) @ vec.T, eig


def learn_response(halves, conditions, floor=1e-8, permutation=None):
    """Learn signal from same-person independent halves, not hidden parameters.

    Each half is a mean over the same number of independent event opportunities.
    Application uses one half's budget. Off-diagonal noise is not assumed zero.
    """
    y, z = np.asarray(halves, float), np.asarray(conditions, float)
    if y.ndim != 3 or y.shape[1:] != (len(z), 2) or len(y) < 3:
        raise ValueError("Need author by condition by two independent halves")
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite source values")
    a, b = y[:, :, 0], y[:, :, 1]
    mean = y.mean(axis=(0, 2))
    noise, noise_eig = project_psd(np.cov(a - b, rowvar=False) / 2)
    noise += floor * np.eye(len(z))
    if permutation is not None:
        permutation = np.asarray(permutation)
        if sorted(permutation.tolist()) != list(range(len(y))):
            raise ValueError("Whole-author permutation required")
        b = b[permutation]
    raw = (a - a.mean(0)).T @ (b - b.mean(0)) / (len(y) - 1)
    kernel, eig = project_psd(raw)
    distances = np.abs(z[:, None] - z)
    width = np.median(distances[distances > 0])
    bases = np.array(
        [np.ones_like(kernel), np.exp(-0.5 * (distances / width) ** 2), np.eye(len(z))]
    )
    coef, residual = nnls(bases.reshape(3, -1).T, ((raw + raw.T) / 2).ravel())
    p = np.eye(len(z)) - np.ones_like(kernel) / len(z)
    return {
        "mean": mean,
        "kernel": kernel,
        "noise": noise,
        "raw_signal": (raw + raw.T) / 2,
        "raw_eigenvalues": eig,
        "noise_eigenvalues": noise_eig,
        "geometry_coefficients": coef,
        "geometry_kernel": np.einsum("k,kij->ij", coef, bases),
        "geometry_residual": float(residual),
        "geometry_width": float(width),
        "response_kernel": p @ kernel @ p,
        "source_authors": len(y),
        "paired": permutation is None,
    }


def compile_function(model, observed, arm="response_covariance"):
    """Freeze one linear person function and its working error covariance.

    The level-only arm predicts the shared mean and only the person's average
    departure. Its uncertainty includes omitted response variation, not zero.
    """
    m = np.asarray(model["mean"], float)
    k = np.asarray(
        model["geometry_kernel"] if arm == "condition_geometry" else model["kernel"],
        float,
    )
    r = np.asarray(model["noise"], float)
    obs = np.asarray(observed, int)
    if (
        len(obs) == 0
        or len(set(obs.tolist())) != len(obs)
        or np.any((obs < 0) | (obs >= len(m)))
    ):
        raise ValueError("Distinct supported observation conditions required")
    s = (k + r)[np.ix_(obs, obs)]
    w = np.linalg.solve(s, k[obs]).T
    if arm == "source_mean":
        w = np.zeros_like(w)
    elif arm == "level_only":
        w = np.ones((len(m), 1)) @ w.mean(axis=0, keepdims=True)
    elif arm not in {"response_covariance", "condition_geometry", "unpaired_source"}:
        raise ValueError("Unknown arm")
    error = k - w @ k[obs] - k[:, obs] @ w.T + w @ s @ w.T
    p = np.eye(len(m)) - np.ones_like(k) / len(m)
    return {
        "observed": obs,
        "weights": w,
        "offset": m - w @ m[obs],
        "mean": m,
        "covariance": (error + error.T) / 2,
        "personal_level_weights": np.ones((len(m), 1)) @ w.mean(axis=0, keepdims=True),
        "personal_response_weights": p @ w,
        "information_resolution": w @ np.eye(len(m))[obs],
    }


def apply_function(program, observations):
    """Apply only observed-condition values and preserve source/personal accounts."""
    y = np.asarray(observations, float)
    w, m = np.asarray(program["weights"]), np.asarray(program["mean"])
    obs = np.asarray(program["observed"], int)
    if y.ndim != 2 or y.shape[1] != len(obs) or not np.isfinite(y).all():
        raise ValueError("Only finite observed-condition means may enter scoring")
    residual = y - m[obs]
    level = residual @ np.asarray(program["personal_level_weights"]).T
    response = residual @ np.asarray(program["personal_response_weights"]).T
    value = y @ w.T + program["offset"]
    np.testing.assert_allclose(value, m + level + response, atol=1e-12)
    return {"value": value, "personal_level": level, "personal_response": response}


def source_neighbors(source, application, kind, rng):
    """One donor diagnostic from observed levels or centered responses, never labels."""
    a, b = np.asarray(source, float), np.asarray(application, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Common observed-condition units required")
    if kind == "random":
        return rng.integers(len(a), size=len(b))
    if kind == "level":
        a, b = a.mean(1, keepdims=True), b.mean(1, keepdims=True)
    elif kind == "response":
        a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    else:
        raise ValueError("Unknown matching rule")
    return np.argmin(np.mean((b[:, None] - a[None]) ** 2, axis=2), axis=1)
