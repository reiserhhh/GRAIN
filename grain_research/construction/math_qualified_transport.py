#!/usr/bin/env python3
"""Common-carrier geometry and marginal-preserving joint-evidence diagnostics."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_text_response import support_frame

SIGNS = np.array([1.0, -1.0, -1.0, 1.0])


def principal_cosines(source, application, rtol=1e-10):
    """Compare full empirical spans, without choosing axes on target outcomes."""
    a, b = np.asarray(source, float), np.asarray(application, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("Columns must share the same ambient carrier coordinates")
    u, _ = support_frame(a, rtol)
    v, _ = support_frame(b, rtol)
    if not u.shape[1] or not v.shape[1]:
        raise ValueError("An empty span has no declared principal-angle comparison")
    return np.clip(np.linalg.svd(u.T @ v, compute_uv=False), 0, 1)


def project_joint_corners(corners, source_basis):
    """Change only the joint four-corner component, preserving both marginals.

    Order is [high/A, low/A, high/B, low/B]. The resulting diagnostic vectors
    need not correspond to actual text and are not cold target predictions.
    """
    x, b = np.asarray(corners, float), np.asarray(source_basis, float)
    if (
        x.ndim != 2
        or x.shape[0] != 4
        or b.ndim != 2
        or b.shape[0] != x.shape[1]
        or not np.isfinite(x).all()
        or not np.isfinite(b).all()
        or not np.allclose(b.T @ b, np.eye(b.shape[1]), atol=1e-10, rtol=0)
    ):
        raise ValueError("Finite aligned corners and orthonormal source basis required")
    mixed = SIGNS @ x
    retained = b @ (b.T @ mixed)
    residual = mixed - retained
    return x - SIGNS[:, None] * residual[None] / 4, retained, residual


def marginal_means(corners):
    """Return the two content and two owner means of the complete design."""
    x = np.asarray(corners, float)
    if x.shape[0] != 4 or not np.isfinite(x).all():
        raise ValueError("Four finite corners required")
    return np.stack(
        ((x[0] + x[2]) / 2, (x[1] + x[3]) / 2, (x[0] + x[1]) / 2, (x[2] + x[3]) / 2)
    )


def vector_cosine(left, right):
    """Describe signed fixed-axis alignment, including explicit zero vectors."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Aligned finite vectors required")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.vdot(a, b) / denominator) if denominator else 0.0


def paired_profile_contrast(readings):
    """Integrate two fixed positive evidence laws with identical marginals."""
    f = np.asarray(readings, float)
    if f.shape[0] != 4 or not np.isfinite(f).all():
        raise ValueError("Four actual observations required")
    plus, minus = (1 + 0.5 * SIGNS) / 4, (1 - 0.5 * SIGNS) / 4
    return np.tensordot(plus - minus, f, axes=(0, 0))
