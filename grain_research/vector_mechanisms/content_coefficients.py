#!/usr/bin/env python3
"""Source-fitted contextual coefficient maps for a fixed native reader basis."""

from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import randomized_svd

from grain_research.vector_mechanisms.position_reader import field, position_design


def author_weights(lengths):
    """Assign equal total mass to each author, regardless of token count."""
    if not lengths or any(int(n) != n or n < 2 for n in lengths):
        raise ValueError("At least two tokens per source author required")
    return np.concatenate([np.full(n, 1 / (len(lengths) * n)) for n in lengths])


def inputs(h, g):
    """Validate row-aligned context and four GRAIN fields without any target."""
    h, g = field(h), np.asarray(g, float)
    if g.shape != (len(h), 4) or not np.isfinite(g).all():
        raise ValueError(
            "Four finite GRAIN values aligned to each context row required"
        )
    return h, g


def moments(x, w):
    """Source-weighted centering/scaling, with explicit constant-column fallback."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    if (
        x.ndim != 2
        or w.shape != (len(x),)
        or np.any(w <= 0)
        or not np.isfinite(x).all()
        or not np.isfinite(w).all()
    ):
        raise ValueError("Finite source features and positive row weights required")
    w = w / w.sum()
    mean = w @ x
    sd = np.sqrt(w @ ((x - mean) ** 2))
    constant = sd < 1e-12
    return mean, np.where(constant, 1, sd), constant


def fit_preprocessor(hidden, grain, rank=32, seed=4300):
    """Fit source-only weighted PCA, projected scales and GRAIN scales once."""
    pairs = [inputs(h, g) for h, g in zip(hidden, grain, strict=True)]
    if not pairs or len({h.shape[1] for h, _ in pairs}) != 1:
        raise ValueError("Aligned source hidden widths required")
    h, g = (np.concatenate([p[k] for p in pairs]) for k in (0, 1))
    w = author_weights([len(p[0]) for p in pairs])
    if not isinstance(rank, int) or not 1 <= rank < min(h.shape):
        raise ValueError("Strictly truncated PCA rank required")
    hc = w @ h
    _, singular, basis = randomized_svd(
        (h - hc) * np.sqrt(w[:, None]),
        n_components=rank,
        n_oversamples=10,
        n_iter=5,
        random_state=seed,
    )
    pc, ps, pconstant = moments((h - hc) @ basis.T, w)
    gc, gs, gconstant = moments(g, w)
    return {
        "h_center": hc,
        "h_basis": basis,
        "h_singular": singular,
        "pc_center": pc,
        "pc_scale": ps,
        "pc_constant": pconstant,
        "g_center": gc,
        "g_scale": gs,
        "g_constant": gconstant,
    }


def raw_design(h, g, model, with_grain):
    """Assemble content/position and optionally source-scaled GRAIN interactions."""
    h, g = inputs(h, g)
    if h.shape[1] != len(model["h_center"]):
        raise ValueError("Input hidden width differs from source PCA")
    x = ((h - model["h_center"]) @ model["h_basis"].T - model["pc_center"]) / model[
        "pc_scale"
    ]
    parts = [position_design(len(h), 3)[:, 1:], x]
    if with_grain:
        z = (g - model["g_center"]) / model["g_scale"]
        parts += [z, (x[:, :, None] * z[:, None, :]).reshape(len(h), -1)]
    return np.column_stack(parts)


def fit_head(hidden, grain, gradients, preprocessor, basis, with_grain, alpha=0.01):
    """Fit source density coefficients; the gradient is a training target only."""
    basis = field(basis)
    if not np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-10) or alpha < 0:
        raise ValueError("Orthonormal frozen basis and nonnegative penalty required")
    pairs = [inputs(h, g) for h, g in zip(hidden, grain, strict=True)]
    if len(pairs) != len(gradients):
        raise ValueError("Source gradient count mismatch")
    ys = []
    for (h, _), j in zip(pairs, gradients, strict=True):
        j = field(j)
        if j.shape != h.shape or basis.shape[1] != h.shape[1]:
            raise ValueError("Source gradient/context/basis layout mismatch")
        ys.append(len(j) * j @ basis.T)
    x = np.concatenate([raw_design(h, g, preprocessor, with_grain) for h, g in pairs])
    y = np.concatenate(ys)
    w = author_weights([len(h) for h, _ in pairs])
    center, scale, constant = moments(x, w)
    x = np.column_stack([np.ones(len(x)), (x - center) / scale])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ (w[:, None] * x) + penalty, x.T @ (w[:, None] * y))
    yp = x @ coef
    energy = float(np.sum(w[:, None] * y**2))
    return dict(preprocessor) | {
        "feature_center": center,
        "feature_scale": scale,
        "feature_constant": constant,
        "weights": coef,
        "basis": basis,
        "with_grain": np.array(with_grain),
        "alpha": np.array(alpha),
    }, {
        "source_coefficient_error": float(np.sum(w[:, None] * (yp - y) ** 2) / energy)
        if energy
        else float("nan"),
        "design_columns": x.shape[1],
        "constant_columns": int(constant.sum()),
        "source_authors": len(pairs),
        "source_tokens": len(y),
    }


def predict_coefficients(h, g, model):
    """Score a new numerical context packet; no gradient/label argument exists."""
    x = raw_design(h, g, model, bool(model["with_grain"]))
    x = np.column_stack(
        [np.ones(len(x)), (x - model["feature_center"]) / model["feature_scale"]]
    )
    a = x @ model["weights"]
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite coefficient prediction")
    return a


def coefficient_field(coefficients, basis):
    """Convert predicted density coefficients back into token-gradient units."""
    a, v = field(coefficients), field(basis)
    if a.shape[1] != len(v):
        raise ValueError("Coefficient/basis rank mismatch")
    return a @ v / len(a)
