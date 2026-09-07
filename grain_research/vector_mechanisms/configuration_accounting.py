#!/usr/bin/env python3
"""Retained vector, score and covariance accounting; no fitting or noise deletion."""

import numpy as np


def retained_paths(configuration, pairing, reference):
    """Split complete author/view configuration vectors, preserving all components."""
    c, p, g = (np.asarray(a, float) for a in (configuration, pairing, reference))
    if (
        c.ndim != 3
        or c.shape != p.shape
        or g.shape != (c.shape[-1],)
        or not all(np.isfinite(a).all() for a in (c, p, g))
    ):
        raise ValueError(
            "Finite matching author/view vectors and frozen covector required"
        )
    remaining = c - p
    scores = {"configuration": c @ g, "pairing": p @ g, "remaining": remaining @ g}
    return remaining, scores


def covariance_accounting(y, configuration, pairing, remaining):
    """Return exact empirical covariance/variance identities without independence."""
    arrays = [np.asarray(a, float) for a in (y, configuration, pairing, remaining)]
    if (
        arrays[0].ndim != 1
        or len(arrays[0]) < 2
        or any(a.shape != arrays[0].shape for a in arrays)
        or not np.isfinite(arrays).all()
    ):
        raise ValueError("Paired finite author scores required")
    data = np.stack(arrays)
    centered = data - data.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / data.shape[1]
    addition = float(np.max(np.abs(arrays[1] - arrays[2] - arrays[3])))
    if addition > 1e-12:
        raise ValueError("Retained source scores do not add")
    return {
        "source_covariances": {
            "configuration": float(cov[0, 1]),
            "pairing": float(cov[0, 2]),
            "remaining": float(cov[0, 3]),
        },
        "variances": {
            "configuration": float(cov[1, 1]),
            "pairing": float(cov[2, 2]),
            "remaining": float(cov[3, 3]),
        },
        "pairing_remaining_covariance": float(cov[2, 3]),
        "source_covariance_identity_error": float(cov[0, 1] - cov[0, 2] - cov[0, 3]),
        "variance_identity_error": float(
            cov[1, 1] - cov[2, 2] - cov[3, 3] - 2 * cov[2, 3]
        ),
        "score_addition_error": addition,
    }


def vector_geometry(configuration, pairing, remaining):
    """Describe native-coordinate vector magnitudes and cancellation, not trait energy."""
    arrays = [np.asarray(a, float) for a in (configuration, pairing, remaining)]
    norms = [np.linalg.norm(a, axis=-1).reshape(-1) for a in arrays]
    product = np.sum(arrays[1] * arrays[2], axis=-1).reshape(-1)
    denominator = norms[1] * norms[2]
    valid = denominator > 0
    return {
        "rms_vector_norm": {
            name: float(np.sqrt(np.mean(n**2)))
            for name, n in zip(
                ("configuration", "pairing", "remaining"), norms, strict=True
            )
        },
        "pairing_remaining_mean_cosine": float(
            np.mean(product[valid] / denominator[valid])
        )
        if np.any(valid)
        else None,
        "nonzero_view_pairs": int(valid.sum()),
        "metric": "native Euclidean coordinates, not invariant under general invertible maps",
    }
