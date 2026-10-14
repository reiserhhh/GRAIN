#!/usr/bin/env python3
"""HM26: source-only scalar potentials with value and directional observations."""

from __future__ import annotations

import numpy as np


def validate(x, v, kind):
    """Validate addressed values (kind0) and directional derivatives (kind1)."""
    x, v, kind = np.asarray(x, float), np.asarray(v, float), np.asarray(kind)
    if (
        x.ndim != 2
        or v.shape != x.shape
        or kind.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(v).all()
        or not np.isin(kind, (0, 1)).all()
        or np.any(np.linalg.norm(v[kind == 1], axis=1) == 0)
    ):
        raise ValueError("finite matched nonzero directional observations required")
    return x, v, kind


def covariance(x, v, kind, y, w, other_kind, length2):
    """Return L_x L_y k for the fixed isotropic Gaussian scalar-potential kernel."""
    x, v, kind = validate(x, v, kind)
    y, w, other_kind = validate(y, w, other_kind)
    if x.shape[1] != y.shape[1] or not np.isfinite(length2) or length2 <= 0:
        raise ValueError("matched dimensions and positive squared length required")
    distance = np.maximum(
        np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2 * x @ y.T,
        0,
    )
    kernel = np.exp(-distance / (2 * length2))
    # These contractions avoid allocating an N-by-M-by-D displacement tensor.
    dv = np.sum(x * v, axis=1)[:, None] - v @ y.T
    dw = x @ w.T - np.sum(y * w, axis=1)[None, :]
    factor = np.ones_like(kernel)
    left, right = kind[:, None] == 1, other_kind[None, :] == 1
    factor = np.where(left & ~right, -dv / length2, factor)
    factor = np.where(~left & right, dw / length2, factor)
    factor = np.where(left & right, v @ w.T / length2 - dv * dw / length2**2, factor)
    return kernel * factor


def source_geometry(x, v):
    """Freeze an origin, scalar RMS unit and median bandwidth from source only."""
    x, v, _ = validate(x, v, np.ones(len(x), int))
    origin = x.mean(axis=0)
    centered = x - origin
    unit = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if unit <= 0:
        raise ValueError("source positions are constant")
    z, direction = centered / unit, v / unit
    d = np.maximum(
        np.sum(z * z, axis=1)[:, None] + np.sum(z * z, axis=1)[None, :] - 2 * z @ z.T,
        0,
    )
    upper = d[np.triu_indices(len(z), 1)]
    positive = upper[upper > 100 * np.finfo(float).eps]
    if not len(positive):
        raise ValueError("source bandwidth is unidentifiable")
    return z, direction, origin, unit, float(np.median(positive))


def fit_potential(x, v, values, derivatives, length2, alpha=1.0, hermite=True):
    """Fit one regularized scalar potential; source output alone supplies the prior."""
    x, v, _ = validate(x, v, np.ones(len(x), int))
    values, derivatives = np.asarray(values, float), np.asarray(derivatives, float)
    if (
        values.shape != (len(x),)
        or derivatives.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(derivatives).all()
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("finite source labels and positive fixed regularizer required")
    prior = float(values.mean())
    if hermite:
        positions, directions = np.vstack((x, x)), np.vstack((v, v))
        kinds = np.r_[np.zeros(len(x), int), np.ones(len(x), int)]
        target = np.r_[values - prior, derivatives]
    else:
        positions, directions = x, v
        kinds, target = np.zeros(len(x), int), values - prior
    gram = covariance(
        positions, directions, kinds, positions, directions, kinds, length2
    )
    diagonal = np.sqrt(np.diag(gram))
    if np.any(diagonal <= 0):
        raise ValueError("zero observation unit")
    normalized = gram / diagonal[:, None] / diagonal[None, :]
    system = (normalized + normalized.T) / 2 + alpha * np.eye(len(gram))
    chol = np.linalg.cholesky(system)
    rhs = target / diagonal
    coefficients = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs)) / diagonal
    return {
        "positions": positions,
        "directions": directions,
        "kinds": kinds,
        "target": target,
        "gram": gram,
        "diagonal": diagonal,
        "coefficients": coefficients,
        "prior": prior,
        "length2": length2,
        "alpha": alpha,
        "system_residual": float(
            np.max(abs((gram + alpha * np.diag(diagonal**2)) @ coefficients - target))
        ),
    }


def predict(model, x, v=None):
    """Evaluate the compiled potential or its addressed directional derivative."""
    x = np.asarray(x, float)
    derivative = v is not None
    v = np.asarray(v, float) if derivative else np.zeros_like(x)
    kinds = np.full(len(x), int(derivative))
    result = (
        covariance(
            x,
            v,
            kinds,
            model["positions"],
            model["directions"],
            model["kinds"],
            model["length2"],
        )
        @ model["coefficients"]
    )
    return result if derivative else result + model["prior"]
