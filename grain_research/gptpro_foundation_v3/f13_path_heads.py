#!/usr/bin/env python3
"""Refit frozen B2/M heads on joint imputed histories, not pseudo posteriors."""

import numpy as np

from .f08_estimators import FEATURES
from .f13_history_sampling import path_counts


def path_head_scores(paths, qualifications, fitted):
    """Return per-path raw predictions and clipped working probabilities."""
    paths = np.asarray(paths)
    counts = path_counts(paths, qualifications)
    gram = np.einsum("si,bs,sj->bij", FEATURES, counts.sum(axis=2), FEATURES)
    rhs = counts[:, :, 1] @ FEATURES + np.asarray(fitted["b2"]["coef"])
    coefficients = np.linalg.solve(gram + np.eye(4), rhs[:, :, None])[:, :, 0]
    state = 2 * paths[:, -2] + paths[:, -1]
    b2 = np.einsum("bi,bi->b", FEATURES[state], coefficients)
    raw = (counts[:, :, 1] + 1) / (counts.sum(axis=2) + 2)
    center = np.asarray(fitted["m"]["mean"])
    right = np.asarray(fitted["m"]["right"])
    projected = center + (raw - center) @ right.T @ right
    m = projected[np.arange(len(paths)), state]
    return {"b2": b2, "m": m}


def summarize_path_scores(raw):
    """Keep clipping and imputation variance separate from parameter uncertainty."""
    raw = np.asarray(raw, float)
    values = np.clip(raw, 1e-6, 1 - 1e-6)
    return {
        "mean": float(values.mean()),
        "history_refit_variance": float(values.var()),
        "mean_raw": float(raw.mean()),
        "clip_count": int(np.count_nonzero(raw != values)),
        "event_variance": float(np.mean(values * (1 - values))),
    }
