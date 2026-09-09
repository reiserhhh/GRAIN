#!/usr/bin/env python3
"""A fixed linear snapshot extension and equal-author paired-state aggregation."""

from __future__ import annotations

import numpy as np


def author_means(records):
    """Average exactly two distinct views per reference author, with paired states."""
    grouped = {}
    for row in records:
        if row["role"] != "reference" or row["view"] not in (0, 1):
            raise ValueError(
                "Only reference-author views0/1 can construct the extension"
            )
        g = grouped.setdefault(row["author_id"], {})
        if row["view"] in g:
            raise ValueError("Duplicate reference view")
        x, y = np.asarray(row["x"], float), np.asarray(row["y"], float)
        if x.ndim != 1 or x.shape != y.shape or not np.isfinite([x, y]).all():
            raise ValueError("Finite matched early/upper states required")
        g[row["view"]] = (x, y)
    if not grouped or any(set(v) != {0, 1} for v in grouped.values()):
        raise ValueError("Every reference author must have both views")
    authors = sorted(grouped)
    x = np.stack(
        [np.mean([v[0] for v in grouped[a].values()], axis=0) for a in authors]
    )
    y = np.stack(
        [np.mean([v[1] for v in grouped[a].values()], axis=0) for a in authors]
    )
    return authors, x, y


def fit_linear(x, y, alpha=1.0):
    """Fit only a centered linear residual kernel, with no hyperparameter search."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or x.shape != y.shape or len(x) < 3 or alpha <= 0:
        raise ValueError("Paired matrices and positive regularization required")
    if not np.isfinite([x, y]).all():
        raise ValueError("Nonfinite construction state")
    mean = x.mean(0)
    scale = float(np.sqrt(np.mean(np.sum((x - mean) ** 2, axis=1))))
    if scale <= 1e-12:
        raise ValueError("No construction variation")
    u = (x - mean) / scale
    residual = y - x
    mean_r = residual.mean(0)
    k = u @ u.T
    col, grand = k.mean(0), float(k.mean())
    kc = k - col[None] - col[:, None] + grand
    lhs = kc + alpha * np.eye(len(x))
    return {
        "mean_x": mean,
        "scale": np.asarray(scale),
        "train_x": u,
        "mean_r": mean_r,
        "col1": col,
        "grand1": np.asarray(grand),
        "alpha": np.asarray(alpha),
        "coef1": np.linalg.solve(lhs, residual - mean_r),
        "effective_df1": np.asarray(np.trace(np.linalg.solve(lhs, kc))),
    }


def predict_linear(model, x):
    """Generate later query states solely from the early input and frozen operator."""
    x = np.asarray(x, float)
    if x.shape != model["mean_x"].shape or not np.isfinite(x).all():
        raise ValueError("Invalid early state")
    u = (x - model["mean_x"]) / float(model["scale"])
    k = u @ model["train_x"].T
    kc = k - k.mean() - model["col1"] + model["grand1"]
    return x + model["mean_r"] + kc @ model["coef1"]
