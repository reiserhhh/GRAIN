#!/usr/bin/env python3
"""Construction-only kernel fitting with endpoint-incidence response penalties."""

import numpy as np


def fit_response_kernel(gram, targets, indices, pairs, *, degree=1, secant=False):
    """Fit residual components using source-only centering and a fixed alpha of1.

    The raw Gram can include held inputs for caching, but all fitted moments and
    targets use indices only. Edges touching a held endpoint are excluded.
    Secants change the row metric; they do not add independent target values.
    """
    gram, targets = np.asarray(gram, float), np.asarray(targets, float)
    ii = np.asarray(indices, int)
    if (
        gram.ndim != 2
        or gram.shape[0] != gram.shape[1]
        or targets.ndim != 2
        or len(targets) != len(gram)
    ):
        raise ValueError("Square Gram and matching target rows required")
    if (
        len(ii) < 3
        or len(set(ii.tolist())) != len(ii)
        or min(ii) < 0
        or max(ii) >= len(gram)
    ):
        raise ValueError("Unique valid construction indices required")
    if (
        degree not in (1, 2)
        or not np.isfinite(gram).all()
        or not np.isfinite(targets[ii]).all()
    ):
        raise ValueError(
            "Finite construction observations and declared kernel required"
        )
    raw = gram[np.ix_(ii, ii)]
    raw_col, raw_grand = raw.mean(0), float(raw.mean())
    centered = raw - raw_col[None, :] - raw_col[:, None] + raw_grand
    scale2 = float(np.trace(centered) / len(ii))
    if scale2 <= 1e-12:
        raise ValueError("No construction input variation")
    k = centered / scale2
    if degree == 2:
        k = k + 0.5 * k**2
    col, grand = k.mean(0), float(k.mean())
    k = k - col[None, :] - col[:, None] + grand
    positions = {int(i): j for j, i in enumerate(ii)}
    edges = [
        (positions[int(a)], positions[int(b)])
        for a, b in pairs
        if int(a) in positions and int(b) in positions
    ]
    d = np.zeros((len(edges), len(ii)))
    for row, (a, b) in enumerate(edges):
        if a == b:
            raise ValueError("Self edge is not an intervention")
        d[row, a], d[row, b] = -1, 1
    value_trace = float(np.trace(k) / len(ii))
    edge_trace = float(np.trace(d @ k @ d.T) / len(d)) if len(d) else 0.0
    weight = (
        value_trace / edge_trace
        if secant and edge_trace > max(1e-12 * value_trace, 1e-20)
        else 0.0
    )
    metric = np.eye(len(ii))
    if weight:
        metric += weight * len(ii) / len(d) * (d.T @ d)
    mean = targets[ii].mean(0)
    # In sum-value convention alpha=1; M1=1 preserves the ordinary intercept.
    coef = np.linalg.solve(
        k + np.linalg.solve(metric, np.eye(len(ii))), targets[ii] - mean
    )
    return {
        "indices": ii,
        "raw_col": raw_col,
        "raw_grand": np.asarray(raw_grand),
        "scale2": np.asarray(scale2),
        "col": col,
        "grand": np.asarray(grand),
        "mean": mean,
        "coef": coef,
        "degree": np.asarray(degree),
        "lambda": np.asarray(weight),
        "edge_count": np.asarray(len(edges)),
        "alpha": np.asarray(1.0),
    }


def predict_response_kernel(model, cross_gram):
    """Predict residual components from raw dots against the full source input list."""
    raw = np.asarray(cross_gram, float)[:, model["indices"]]
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Finite query-by-source Gram required")
    centered = raw - raw.mean(1)[:, None] - model["raw_col"] + model["raw_grand"]
    k = centered / model["scale2"]
    if int(model["degree"]) == 2:
        k = k + 0.5 * k**2
    k = k - k.mean(1)[:, None] - model["col"] + model["grand"]
    return model["mean"] + k @ model["coef"]
