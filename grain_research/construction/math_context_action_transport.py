#!/usr/bin/env python3
"""Fixed source calibration of conditional author-account actions, not traits."""

from __future__ import annotations

import numpy as np


def person_channels(readings):
    """Map the actual A6/B6/A-fact/B-fact query order to actor by seven queries."""
    y = np.asarray(readings, float)
    if y.shape[0] != 14 or not np.isfinite(y).all():
        raise ValueError("Fourteen fixed native query readings required")
    return np.stack(
        (np.concatenate([y[:6], y[12:13]]), np.concatenate([y[6:12], y[13:14]]))
    )


def interaction(corners):
    """Return the positive-law contrast on complete b/c support."""
    x = np.asarray(corners, float)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete finite b/c corners required")
    return (x[0, 0] - x[0, 1] - x[1, 0] + x[1, 1]) / 4


def positive_laws():
    """Keep both condition and behavior margins and strictly positive masses."""
    bc = np.outer([-1.0, 1.0], [-1.0, 1.0])
    return (1 + 0.5 * bc) / 4, (1 - 0.5 * bc) / 4


def balanced_labels(n, seed):
    """One outcome-independent sham split, fixed before source/application reads."""
    if n < 2 or n % 2:
        raise ValueError("Positive even support required")
    return np.random.default_rng(seed).permutation(np.repeat([0, 1], n // 2))


def features(g, context=None):
    """Preserve event values; only the supplied context selects a feature block."""
    x = np.asarray(g, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite event coordinate matrix required")
    if context is None:
        return x
    c = np.asarray(context)
    if c.shape != (len(x),) or not np.isin(c, [0, 1]).all():
        raise ValueError("Observed binary context required, never missing-as-zero")
    return (np.eye(2)[c.astype(int), :, None] * x[:, None, :]).reshape(len(x), -1)


def fit_actions(x, y, rtol=1e-10):
    """One source-RMS minimum-norm calibration; no held labels or tuning."""
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b):
        raise ValueError("Aligned source actions required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite source actions required")
    scale = np.sqrt(np.mean(a * a, axis=0))
    scale[scale == 0] = 1
    z = a / scale
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    keep = s > (s[0] * rtol if len(s) else 0)
    normalized = (vt[keep].T / s[keep]) @ (u[:, keep].T @ b)
    beta = normalized / scale[:, None]
    alt = np.linalg.lstsq(z, b, rcond=rtol)[0] / scale[:, None]
    return {
        "beta": beta.tolist(),
        "scale": scale.tolist(),
        "rank": int(keep.sum()),
        "singular_values": s.tolist(),
        "source_error_rms": float(np.sqrt(np.mean((a @ beta - b) ** 2))),
        "alternate_prediction_error": float(np.max(abs(a @ (beta - alt)))),
        "intercept": 0.0,
        "unit_scope": "contrast_calibrated_native_observer_not_human_trait",
    }


def cold_score(bundle, records):
    """Pure numeric scoring; reject target/design/corner lookup fields."""
    allowed = {"g", "real_context", "sham_context"}
    if not records or any(set(r) != allowed for r in records):
        raise ValueError("Cold record must contain only coordinate and contexts")
    g = np.array([r["g"] for r in records], float)
    if g.shape[1:] != (15,):
        raise ValueError("All15 frozen event coordinates required")
    real = np.array([r["real_context"] for r in records])
    sham = np.array([r["sham_context"] for r in records])
    dictionaries = {
        "global": features(g),
        "real": features(g, real),
        "sham": features(g, sham),
        "wrong_real": features(g, 1 - real),
    }
    return {
        name: x
        @ np.asarray(bundle["models"]["real" if name == "wrong_real" else name]["beta"])
        for name, x in dictionaries.items()
    }


def compare(predicted, observed):
    """Report every signed action, preserving undefined zero-normalized metrics."""
    p, y = np.asarray(predicted, float), np.asarray(observed, float)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Aligned finite actions required")
    pp, yy = float(p @ p), float(y @ y)
    py = float(p @ y)
    return {
        "n": len(y),
        "observed_rms": float(np.sqrt(np.mean(y * y))),
        "predicted_rms": float(np.sqrt(np.mean(p * p))),
        "error_rms": float(np.sqrt(np.mean((p - y) ** 2))),
        "relative_error_to_zero": float(np.sqrt(np.sum((p - y) ** 2) / yy))
        if yy
        else None,
        "cosine": py / np.sqrt(pp * yy) if pp and yy else None,
        "signed_projection": py / yy if yy else None,
        "positive_product_count": int(np.sum(p * y > 0)),
        "bias": float(np.mean(p - y)),
    }
