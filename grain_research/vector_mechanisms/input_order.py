#!/usr/bin/env python3
"""Reversal quotient accounting and a fixed order-sensitive input extension."""

import numpy as np
from sklearn.linear_model import Ridge


def ordinal_vector(bag):
    """Return an odd ordinal first moment without changing comment embeddings."""
    x = np.asarray(bag, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite comment vectors required")
    return np.linspace(-1, 1, len(x)) @ x / len(x)


def fit_ordinal(bags, targets, baseline, penalty=0.01):
    """Fit only source teacher errors; zero intercept preserves odd symmetry."""
    x = np.array([ordinal_vector(b) for b in bags])
    y = np.asarray(targets, float) - baseline.predict(bags, None)
    if not np.isfinite(y).all() or penalty <= 0:
        raise ValueError("Finite matched targets and positive penalty required")
    scale = max(float(np.sqrt(np.mean(np.sum(x * x, axis=1)))), 1e-12)
    head = Ridge(alpha=penalty * len(x), fit_intercept=False).fit(x / scale, y)
    return {"baseline": baseline, "scale": scale, "coef": head.coef_}


def predict_ordinal(model, bags):
    """Evaluate a frozen base-plus-odd function on new ordered material."""
    x = np.array([ordinal_vector(b) for b in bags])
    return model["baseline"].predict(bags, None) + x @ model["coef"].T / model["scale"]


def reversal_account(original, reversed_, prediction):
    """Exact orbit-average squared loss for a reversal-invariant prediction."""
    a, b, p = [np.asarray(x, float) for x in (original, reversed_, prediction)]
    if a.shape != b.shape or p.shape != a.shape or not np.isfinite([a, b, p]).all():
        raise ValueError("Matched finite scalar observations required")
    even, odd = (a + b) / 2, (a - b) / 2
    mse = float(np.mean(((a - p) ** 2 + (b - p) ** 2) / 2))
    even_error, odd_floor = float(np.mean((even - p) ** 2)), float(np.mean(odd**2))
    return {
        "pair_mse": mse,
        "even_error": even_error,
        "odd_floor": odd_floor,
        "closure_error": abs(mse - even_error - odd_floor),
        "delta_mean": float(np.mean(b - a)),
        "delta_sd": float(np.std(b - a)),
        "delta_rms": float(np.sqrt(np.mean((b - a) ** 2))),
        "odd_common_energy": float(odd.mean() ** 2),
        "odd_author_energy": float(odd.var()),
    }
