#!/usr/bin/env python3
"""Source-compiled, integrable GRAIN/query moment observations.

The new quadratic potential uses within-exact-GRAIN-cell state covariance.
Source query gradients define a frozen subspace; application needs only current
states/GRAIN coordinates, not current gradients, target scores, or labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import randomized_svd


def cell_residual(g, h):
    """Remove exact-cell means for a separate statistic, never delete raw input."""
    g, h = np.asarray(g, float), np.asarray(h, float)
    if g.ndim != 2 or h.ndim != 2 or len(g) != len(h) or len(g) < 2:
        raise ValueError("Aligned token matrices required")
    if not np.isfinite(g).all() or not np.isfinite(h).all():
        raise ValueError("Finite matrices required")
    _, index, counts = np.unique(g, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), h.shape[1]))
    np.add.at(sums, index, h)
    return h - (sums / counts[:, None])[index]


def polynomial(g, mean, scale, degree):
    """Frozen-source GRAIN coordinates; complete degree two, no feature selection."""
    g = np.asarray(g, float)
    mean, scale = np.asarray(mean), np.asarray(scale)
    if g.ndim != 2 or g.shape[1] != 4 or mean.shape != (4,) or scale.shape != (4,):
        raise ValueError("Four aligned GRAIN coordinates required")
    if degree not in (1, 2) or np.any(scale <= 0) or not np.isfinite(g).all():
        raise ValueError("Degree 1/2 and finite, nondegenerate coordinates required")
    z = (g - mean) / scale
    terms = [np.ones(len(z)), *z.T]
    if degree == 2:
        terms.extend(z[:, i] * z[:, j] for i in range(4) for j in range(i, 4))
    return np.column_stack(terms)


def symmetric_basis(rank):
    """Orthonormal basis of symmetric matrices, including signed off-diagonals."""
    if rank < 1:
        raise ValueError("Positive subspace rank required")
    matrices = []
    for i in range(rank):
        for j in range(i, rank):
            matrix = np.zeros((rank, rank))
            matrix[i, j] = 1 if i == j else 1 / np.sqrt(2)
            if i != j:
                matrix[j, i] = 1 / np.sqrt(2)
            matrices.append(matrix)
    return np.stack(matrices)


def ridge_normal(gram, rhs, alpha):
    """Solve one scale-relative regularized normal equation; no held tuning."""
    penalty = alpha * max(float(np.trace(gram) / len(gram)), 1e-12)
    return np.linalg.solve(gram + penalty * np.eye(len(gram)), rhs)


def fit_operator(panels, rank=4, alpha=1e-6, seed=91351):
    """Compile query fields from SOURCE authors, with equal total weight per author.

    Each panel provides g, h, j and scalar value from the same source context.
    Linear/quadratic fields depend only on g. The symmetric within-cell potential
    has an explicit scalar primitive, avoiding a non-integrable arbitrary field.
    """
    if len(panels) < 2 or alpha <= 0:
        raise ValueError("At least two source panels and positive penalty required")
    g_all = np.concatenate([p["g"] for p in panels])
    weights = np.concatenate(
        [np.full(len(p["g"]), 1 / (len(panels) * len(p["g"]))) for p in panels]
    )
    mean = weights @ g_all
    scale = np.sqrt(weights @ ((g_all - mean) ** 2))
    if np.any(scale < 1e-10):
        raise ValueError("Degenerate source GRAIN coordinate")
    fields, residuals = [], []
    for p in panels:
        if p["h"].shape != p["j"].shape or len(p["h"]) != len(p["g"]):
            raise ValueError("Source state/gradient addresses differ")
        n = len(p["h"])
        field = n * np.asarray(p["j"], float)
        if not np.isfinite(field).all():
            raise ValueError("Nonfinite source field")
        fields.append(field)
        residuals.append(cell_residual(p["g"], field) / np.sqrt(n * len(panels)))
    _, singular, vt = randomized_svd(
        np.concatenate(residuals), n_components=rank, n_iter=3, random_state=seed
    )
    v = vt.T
    model = {
        "g_mean": mean,
        "g_scale": scale,
        "query_basis": v,
        "source_singular": singular,
        "alpha": np.array(alpha),
    }
    target = np.concatenate(fields)
    for degree in (1, 2):
        x = polynomial(g_all, mean, scale, degree)
        model[f"beta{degree}"] = ridge_normal(
            x.T @ (weights[:, None] * x), x.T @ (weights[:, None] * target), alpha
        )
    sb = symmetric_basis(rank)
    gram, rhs = np.zeros((len(sb), len(sb))), np.zeros(len(sb))
    for p, field in zip(panels, fields):
        n = len(p["h"])
        r = cell_residual(p["g"], p["h"]) @ v
        target_r = cell_residual(p["g"], field) @ v
        design = 2 * np.einsum("tr,krs->tsk", r, sb).reshape(-1, len(sb))
        weight = 1 / (n * len(panels))
        gram += weight * design.T @ design
        rhs += weight * design.T @ target_r.reshape(-1)
    model["within_matrix"] = np.einsum("k,kij->ij", ridge_normal(gram, rhs, alpha), sb)
    # Derivatives do not identify scalar offsets; anchor levels on source only.
    offsets = []
    for p in panels:
        value, _ = apply_operator(p["g"], p["h"], model, anchor=False)
        offsets.append(float(p["value"]) - value)
    model["offsets"] = np.mean(offsets, axis=0)
    return model


def apply_operator(g, h, model, *, anchor=True):
    """Return three nested scalar readers and exact state gradients.

    GRAIN is held fixed in this derivative, matching the declared hidden-state
    intervention. Raw text changes may also change GRAIN and token membership.
    """
    g, h = np.asarray(g, float), np.asarray(h, float)
    if g.ndim != 2 or h.ndim != 2 or len(g) != len(h) or not np.isfinite(h).all():
        raise ValueError("Aligned finite inputs required")
    n = len(h)
    fields, values = [], []
    for degree in (1, 2):
        field = (
            polynomial(g, model["g_mean"], model["g_scale"], degree)
            @ model[f"beta{degree}"]
            / n
        )
        fields.append(field)
        values.append(np.sum(field * h))
    v, a = model["query_basis"], model["within_matrix"]
    residual = cell_residual(g, h)
    r = residual @ v
    correction = float(np.einsum("ti,ij,tj->", r, a, r) / n)
    correction_field = 2 * (r @ a @ v.T) / n
    values.append(values[1] + correction)
    fields.append(fields[1] + correction_field)
    values = np.asarray(values)
    if anchor:
        values += model["offsets"]
    return values, np.stack(fields)


def observation_moments(g, h, model):
    """Expose mean/interaction moments and new within-cell query covariance."""
    g, h = np.asarray(g, float), np.asarray(h, float)
    r = cell_residual(g, h) @ model["query_basis"]
    return {
        "linear": polynomial(g, model["g_mean"], model["g_scale"], 1).T @ h / len(h),
        "quadratic": polynomial(g, model["g_mean"], model["g_scale"], 2).T @ h / len(h),
        "within_covariance": r.T @ r / len(h),
    }


def within_direction(g, h, basis):
    """Make a current-input-only direction invisible to all exact-g mean moments."""
    raw = (cell_residual(g, h) @ basis) @ basis.T
    norm = float(np.sqrt(np.mean(raw**2)))
    if norm < 1e-15:
        raise ValueError("No within-cell variation in this frozen subspace")
    return raw * (float(np.sqrt(np.mean(h**2))) / norm)
