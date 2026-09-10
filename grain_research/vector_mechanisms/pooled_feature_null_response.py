#!/usr/bin/env python3
"""Native directional responses visible and invisible to BODY-mean PCA features."""

from __future__ import annotations

from itertools import combinations

import numpy as np

PARTS = ("visible", "allocation", "outside")


def decompose(direction, basis):
    """Orthogonally split a BODY direction in its declared Euclidean metric."""
    d, u = np.asarray(direction, float), np.asarray(basis, float)
    if (
        d.ndim != 2
        or len(d) < 2
        or u.ndim != 2
        or u.shape[0] != d.shape[1]
        or not 0 < u.shape[1] <= u.shape[0]
        or not np.isfinite(d).all()
        or not np.isfinite(u).all()
    ):
        raise ValueError("Finite BODY-by-feature directions and aligned basis required")
    orth = float(np.max(np.abs(u.T @ u - np.eye(u.shape[1]))))
    if orth > 1e-6:
        raise ValueError("Source basis must be orthonormal")
    mean = d.mean(axis=0)
    visible = np.broadcast_to((mean @ u) @ u.T, d.shape)
    parts = (visible, d - mean, np.broadcast_to(mean - visible[0], d.shape))
    total_energy = float(np.sum(d * d))
    geom = {
        "orthogonality_error": orth,
        "total_energy": total_energy,
        "reconstruction_error": float(np.linalg.norm(sum(parts) - d)),
        "feature_derivative_norm": float(np.linalg.norm(mean @ u)),
    }
    for name, part in zip(PARTS, parts, strict=True):
        geom[name + "_energy"] = float(np.sum(part * part))
        expected = mean @ u if name == "visible" else np.zeros(u.shape[1])
        geom[name + "_feature_error"] = float(
            np.linalg.norm(part.mean(axis=0) @ u - expected)
        )
    for (na, a), (nb, b) in combinations(zip(PARTS, parts, strict=True), 2):
        geom[f"{na}_{nb}_inner"] = float(np.sum(a * b))
    geom["energy_identity_error"] = abs(
        sum(geom[n + "_energy"] for n in PARTS) - total_energy
    )
    return parts, geom


def responses(gradient, direction, basis):
    """Contract all components with the same original BODY gradient."""
    g, d = np.asarray(gradient, float), np.asarray(direction, float)
    if g.shape != d.shape or not np.isfinite(g).all():
        raise ValueError("Finite gradient at the exact BODY addresses required")
    parts, geom = decompose(d, basis)
    values = {n: float(np.sum(g * p)) for n, p in zip(PARTS, parts, strict=True)}
    values["native"] = float(np.sum(g * d))
    values["null"] = values["allocation"] + values["outside"]
    values["sum_error"] = abs(sum(values[n] for n in PARTS) - values["native"])
    return values, geom


def response_account(values):
    """Summarize signed action channels, keeping covariance/cancellation explicit."""
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != 4 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Rows of native, visible, allocation, outside required")
    names = ("native", *PARTS, "null")
    x = np.column_stack((x, x[:, 2] + x[:, 3]))
    energy = float(x[:, 0] @ x[:, 0])
    result = {"rows": len(x), "native_squared_sum": energy}
    for i, name in enumerate(names):
        result[name + "_mean"] = float(x[:, i].mean())
        result[name + "_rms"] = float(np.sqrt(np.mean(x[:, i] ** 2)))
        result[name + "_projection"] = (
            float(x[:, i] @ x[:, 0] / energy) if energy > 1e-24 else np.nan
        )
        result[name + "_squared_ratio"] = (
            float(x[:, i] @ x[:, i] / energy) if energy > 1e-24 else np.nan
        )
    for (i, a), (j, b) in combinations(enumerate(PARTS, start=1), 2):
        result[f"cross_{a}_{b}"] = float(2 * np.mean(x[:, i] * x[:, j]))
    predicted_energy = sum(result[n + "_rms"] ** 2 for n in PARTS)
    predicted_energy += sum(v for k, v in result.items() if k.startswith("cross_"))
    result["response_energy_identity_error"] = abs(
        predicted_energy - result["native_rms"] ** 2
    )
    return result
