#!/usr/bin/env python3
"""Finite target observability of actual frozen GRAIN moment summaries."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr

ARMS = (
    "pooled_mean",
    "pooled_diag",
    "pooled_full",
    "panel_diag",
    "embedding_mean",
    "rotated_diag",
    "rotated_full",
)
ROTATION = (
    np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], float) / 2
)


def _condition_columns(values):
    """Change the moment basis without changing its observable row space."""
    x = np.asarray(values, float)
    center, scale = x.mean(0), x.std(0)
    return np.divide(x - center, scale, out=np.zeros_like(x), where=scale > 0)


def design(grain, panel, arm, embedding=None):
    """Construct fixed summary rows; panels retain their exact probability mass."""
    g, p = np.asarray(grain, float), np.asarray(panel)
    if g.ndim != 2 or g.shape[1] != 4 or p.shape != (len(g),) or len(g) < 2:
        raise ValueError("Aligned four-coordinate comment support required")
    if not np.isfinite(g).all() or arm not in ARMS:
        raise ValueError("Finite values and declared arm required")
    indicator = (p[:, None] == np.unique(p)[None, :]).astype(float)
    if arm == "embedding_mean":
        h = np.asarray(embedding, float)
        if h.ndim != 2 or len(h) != len(g) or not np.isfinite(h).all():
            raise ValueError("Aligned finite embedding required")
        return np.column_stack((indicator, _condition_columns(h))).T
    if arm.startswith("rotated"):
        g = g @ ROTATION
    z = _condition_columns(g)
    if arm == "panel_diag":
        parts = [indicator]
        for i in range(indicator.shape[1]):
            parts.extend((indicator[:, i, None] * z, indicator[:, i, None] * z**2))
    else:
        parts = [indicator, z]
        if arm != "pooled_mean":
            parts.append(z**2)
        if arm.endswith("full"):
            parts.append(
                np.column_stack(
                    [z[:, i] * z[:, j] for i in range(4) for j in range(i + 1, 4)]
                )
            )
    return np.column_stack(parts).T


def row_basis(matrix):
    """Use a frozen machine convention, independently checking with pivoted QR."""
    a = np.asarray(matrix, float)
    if a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError("Nonempty finite summary design required")
    _, singular, vt = np.linalg.svd(a, full_matrices=False)
    tol = 100 * np.finfo(float).eps * max(a.shape) * singular[0]
    rank = int(np.sum(singular > tol))
    basis = vt[:rank].T
    q, _, _ = qr(a.T, mode="economic", pivoting=True)
    return basis, q[:, :rank], rank, float(tol)


def witness(matrix, observations, radius=0.5):
    """Produce strictly positive same-summary distributions, not fitted scores."""
    a, y = np.asarray(matrix, float), np.asarray(observations, float)
    if (
        a.ndim != 2
        or y.ndim != 2
        or y.shape[1] == 0
        or len(y) != a.shape[1]
        or not np.isfinite(y).all()
    ):
        raise ValueError("Finite comment-by-target observations required")
    if not 0 < radius < 1:
        raise ValueError("Strict interior radius required")
    b, q, rank, tol = row_basis(a)
    r = y - b @ (b.T @ y)
    rq = y - q @ (q.T @ y)
    qr_error = float(np.max(np.abs(r - rq)))
    if qr_error > 5e-8:
        raise ArithmeticError("Alternate QR/SVD row-space disagreement")
    largest = np.max(np.abs(r), axis=0)
    active = largest > 1e-10 * np.maximum(1, np.linalg.norm(y, axis=0))
    delta = np.divide(r, largest, out=np.zeros_like(r), where=active) * radius / len(y)
    plus, minus = 1 / len(y) + delta, 1 / len(y) - delta
    mass_error = float(max(np.max(abs(plus.sum(0) - 1)), np.max(abs(minus.sum(0) - 1))))
    summary_error = float(np.max(abs(a @ (plus - minus))))
    if mass_error > 1e-9 or summary_error > 1e-8 or min(plus.min(), minus.min()) <= 0:
        raise ArithmeticError("Same-summary probability witness failed")
    yc = y - y.mean(0)
    power = np.sum(yc**2, axis=0)
    energy = np.sum(r**2, axis=0)
    fraction = np.divide(energy, power, out=np.zeros_like(energy), where=power > 1e-20)
    return {
        "residual": r,
        "plus": plus,
        "minus": minus,
        "gap": np.sum((plus - minus) * y, axis=0),
        "residual_energy_fraction": fraction,
        "active": active,
        "rank": rank,
        "tolerance": tol,
        "null_dimension": a.shape[1] - rank,
        "qr_error": qr_error,
        "mass_error": mass_error,
        "summary_error": summary_error,
    }
