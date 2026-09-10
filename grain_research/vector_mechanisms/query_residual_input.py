#!/usr/bin/env python3
"""Equal-width variance and query-linked residual extensions of a frozen reader."""

from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import randomized_svd

from grain_research.vector_mechanisms.content_coefficients import (
    author_weights,
    inputs,
    moments,
    predict_coefficients,
    raw_design,
)


def residual_hidden(h, base):
    """Remove only the frozen input PCA projection, leaving its complement intact."""
    h = np.asarray(h, float) - base["h_center"]
    p = base["h_basis"]
    return h - (h @ p.T) @ p


def base_design(h, g, base):
    """Exactly preserve the earlier source-standardized36 columns, including bias."""
    if bool(base["with_grain"]):
        raise ValueError("Only the frozen content-only base is permitted")
    x = raw_design(h, g, base, False)
    return np.column_stack(
        [np.ones(len(x)), (x - base["feature_center"]) / base["feature_scale"]]
    )


def complement_rows(rows, p):
    """Reproject/orthonormalize with a fixed largest-coordinate-positive sign."""
    rows = np.asarray(rows, float)
    if not len(rows):
        return rows.copy()
    r = rows - (rows @ p.T) @ p
    r -= (r @ p.T) @ p
    q, triangle = np.linalg.qr(r.T, mode="reduced")
    if np.min(abs(np.diag(triangle))) < 1e-10:
        raise ValueError("Residual directions lost numerical rank after projection")
    r = q.T
    signs = np.sign(r[np.arange(len(r)), np.argmax(abs(r), axis=1)])
    r *= signs[:, None]
    if np.max(abs(r @ p.T)) > 1e-10 or np.max(abs(r @ r.T - np.eye(len(r)))) > 1e-10:
        raise ValueError("Residual direction orthogonality failed")
    return r


def variance_basis(hidden, base, rank=8, seed=4400):
    """Fit the same-sized residual variance control with source-author weights."""
    r = np.concatenate([residual_hidden(h, base) for h in hidden])
    w = author_weights([len(h) for h in hidden])
    _, s, v = randomized_svd(
        r * np.sqrt(w[:, None]),
        n_components=rank,
        n_oversamples=10,
        n_iter=5,
        random_state=seed,
    )
    return {"extra_basis": complement_rows(v, base["h_basis"]), "extra_singular": s}


def covariance_basis(residual, error, w, p, rank=8):
    """Select source cross-covariance directions, not held outcomes or input energy."""
    r, e, w = (
        np.asarray(residual, float),
        np.asarray(error, float),
        np.asarray(w, float),
    )
    if (
        r.ndim != 2
        or e.ndim != 2
        or len(r) != len(e)
        or w.shape != (len(r),)
        or np.any(w <= 0)
        or not np.isfinite(r).all()
        or not np.isfinite(e).all()
    ):
        raise ValueError(
            "Aligned finite source residuals and positive weights required"
        )
    c = r.T @ (w[:, None] * e)
    if np.max(abs(c)) == 0:
        return {"extra_basis": np.zeros((0, r.shape[1])), "extra_singular": np.zeros(0)}
    u, s, _ = np.linalg.svd(c, full_matrices=False)
    count = min(rank, int(np.sum(s > 1e-10 * s[0])))
    return {"extra_basis": complement_rows(u[:, :count].T, p), "extra_singular": s}


def fit_extension(hidden, grain, gradients, base, extra=None, alpha=0.01):
    """Fit one equal-author density head using an independently chosen residual span."""
    pairs = [inputs(h, g) for h, g in zip(hidden, grain, strict=True)]
    if len(pairs) != len(gradients):
        raise ValueError("Source gradient count mismatch")
    ys = []
    for (h, _), j in zip(pairs, gradients, strict=True):
        j = np.asarray(j, float)
        if j.shape != h.shape or not np.isfinite(j).all():
            raise ValueError("Source gradient layout mismatch")
        ys.append(len(h) * j @ base["basis"].T)
    y = np.concatenate(ys)
    r = np.concatenate([residual_hidden(h, base) for h, _ in pairs])
    w = author_weights([len(h) for h, _ in pairs])
    phi = np.concatenate([base_design(h, g, base) for h, g in pairs])
    if extra is None:
        old = np.concatenate([predict_coefficients(h, g, base) for h, g in pairs])
        extra = covariance_basis(r, y - old, w, base["h_basis"])
    v = extra["extra_basis"]
    z = r @ v.T
    center, scale, constant = moments(z, w)
    z = (z - center) / scale
    x = np.column_stack([phi, z])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ (w[:, None] * x) + penalty, x.T @ (w[:, None] * y))
    model = (
        dict(base)
        | extra
        | {
            "extra_center": center,
            "extra_scale": scale,
            "extra_constant": constant,
            "extra_source_min": z.min(0),
            "extra_source_max": z.max(0),
            "aug_weights": coef,
        }
    )
    retained = float(np.sum(w[:, None] * (r @ v.T) ** 2))
    total = float(np.sum(w[:, None] * r**2))
    return model, {
        "extra_rank": len(v),
        "design_columns": x.shape[1],
        "source_coefficient_error": float(
            np.sum(w[:, None] * (y - x @ coef) ** 2) / np.sum(w[:, None] * y**2)
        ),
        "extra_residual_energy_fraction": retained / total if total else 0.0,
        "pca_overlap_max": float(np.max(abs(v @ base["h_basis"].T))) if len(v) else 0.0,
        "source_coordinate_abs_max": float(np.max(abs(z))) if len(v) else 0.0,
    }


def predict_extension(h, g, model):
    """Use new author H/G and fixed numerical assets only; never read a gradient."""
    x = base_design(h, g, model)
    z = (
        residual_hidden(h, model) @ model["extra_basis"].T - model["extra_center"]
    ) / model["extra_scale"]
    result = np.column_stack([x, z]) @ model["aug_weights"]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite residual coefficient prediction")
    return result


def coordinate_support(h, model):
    """Describe held coordinates against source extrema without changing prediction."""
    z = (
        residual_hidden(h, model) @ model["extra_basis"].T - model["extra_center"]
    ) / model["extra_scale"]
    return {
        "held_coordinate_abs_max": float(np.max(abs(z))) if z.size else 0.0,
        "held_coordinate_outside_fraction": float(
            np.mean((z < model["extra_source_min"]) | (z > model["extra_source_max"]))
        )
        if z.size
        else 0.0,
    }
