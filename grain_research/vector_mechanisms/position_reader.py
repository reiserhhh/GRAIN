#!/usr/bin/env python3
"""Separate a hidden-direction basis from token-position reading coefficients."""

from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import randomized_svd


def field(x):
    """Validate one scalar-output token by hidden gradient field."""
    a = np.asarray(x, float)
    if a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all():
        raise ValueError("Finite token-by-hidden gradient matrix required")
    return a


def hidden_basis(fields, rank=8, seed=2501):
    """Fit a fixed-rank basis for equally weighted author gradient densities.

    K_u=T_u J_u; squared loss has weight1/T_u per token. Stacking sqrt(T_u)
    times J_u implements that convention without assigning extra mass to long
    authors. This approximates matrix energy, not GRAIN directional actions.
    """
    arrays = [field(x) for x in fields]
    if not arrays or len({x.shape[1] for x in arrays}) != 1:
        raise ValueError("Aligned source hidden widths required")
    x = np.concatenate([np.sqrt(len(a) / len(arrays)) * a for a in arrays])
    if not isinstance(rank, int) or not 1 <= rank < min(x.shape):
        raise ValueError("Strictly truncated positive rank required")
    _, s, v = randomized_svd(
        x, n_components=rank, n_oversamples=10, n_iter=5, random_state=seed
    )
    if np.max(abs(v @ v.T - np.eye(rank))) > 1e-10:
        raise ValueError("Numerical hidden basis is not orthonormal")
    return {
        "basis": v,
        "singular_values": s,
        "weighted_energy": np.array(np.sum(x * x)),
    }


def project_field(jacobian, basis, rank):
    """Oracle token coefficients for a frozen orthonormal hidden-direction basis."""
    j = field(jacobian)
    v = np.asarray(basis, float)
    if (
        v.ndim != 2
        or not np.isfinite(v).all()
        or j.shape[1] != v.shape[1]
        or not 1 <= rank <= len(v)
    ):
        raise ValueError("Rank/hidden layout mismatch")
    v = v[:rank]
    return (j @ v.T) @ v


def position_design(count, degree):
    """Use midpoint-normalized selected-body ordinal positions, not absolute tokens."""
    if not isinstance(count, int) or count < 2 or degree not in (0, 3):
        raise ValueError("At least two tokens and frozen degree0 or3 required")
    x = 2 * (np.arange(count) + 0.5) / count - 1
    return np.polynomial.legendre.legvander(x, degree)


def fit_position(fields, degree):
    """Fit vector-valued position density with equal source-author weight only."""
    arrays = [field(x) for x in fields]
    if not arrays or len({x.shape[1] for x in arrays}) != 1:
        raise ValueError("Aligned source gradient fields required")
    xx, xy = None, None
    for j in arrays:
        x = position_design(len(j), degree)
        gram = x.T @ x / len(j)
        rhs = x.T @ j  # x.T @ (T*j) / T
        xx = gram if xx is None else xx + gram
        xy = rhs if xy is None else xy + rhs
    if np.linalg.matrix_rank(xx) != degree + 1:
        raise ValueError("Source positions do not identify polynomial coefficients")
    return {
        "coefficients": np.linalg.solve(xx, xy),
        "degree": np.array(degree),
        "source_authors": np.array(len(arrays)),
    }


def predict_position(count, model):
    """Read a target length and fixed coefficients only; no target gradient/H/G."""
    x = position_design(count, int(model["degree"]))
    coef = np.asarray(model["coefficients"], float)
    if coef.ndim != 2 or coef.shape[0] != x.shape[1] or not np.isfinite(coef).all():
        raise ValueError("Finite source coefficients matching degree required")
    return x @ coef / count


def contractions(jacobian, directions):
    """Contract a reader with explicitly aligned GRAIN perturbation directions."""
    j = field(jacobian)
    d = np.asarray(directions, float)
    if d.ndim != 3 or d.shape[1:] != j.shape or not np.isfinite(d).all():
        raise ValueError("Factor-by-token-by-hidden directions required")
    return np.einsum("th,kth->k", j, d)


def errors(actual, predicted, directions):
    """Return field energy and action error separately, including the Cauchy bound."""
    j, p = field(actual), field(predicted)
    if p.shape != j.shape or not np.sum(j * j):
        raise ValueError("Aligned nonzero target field required")
    y, yp = contractions(j, directions), contractions(p, directions)
    residual = p - j
    norm = np.linalg.norm(residual)
    bound = norm * np.linalg.norm(np.asarray(directions).reshape(len(y), -1), axis=1)
    if np.any(abs(yp - y) > bound + 1e-10):
        raise ValueError("Directional error exceeds Frobenius Cauchy bound")
    return {
        "field_error_ratio": float(np.sum(residual**2) / np.sum(j**2)),
        "field_actual_energy": float(np.sum(j * j)),
        "field_error_energy": float(np.sum(residual**2)),
        "actual": y,
        "predicted": yp,
        "error_bound": bound,
    }
