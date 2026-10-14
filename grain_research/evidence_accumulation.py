#!/usr/bin/env python3
"""Condition-response scores with explicit dependent-event accounting.

These estimators inherit weighted least squares and cluster sandwich covariance.
Known event identities are an observation design input, not inferred psychological
truth. No covariance correction can by itself repair selective over-reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


def profile_fit(condition, evidence, event_ids, *, balance_events, cluster_errors):
    """Fit level and response; distinguish report weights from event weights.

    Covariance is a small-sample-adjusted sandwich, conditional on the supplied
    evidence/condition reader. It does not include uncertainty in that reader.
    Returns None for an unidentifiable profile rather than a prior-filled score.
    """
    z, y = np.asarray(condition, float), np.asarray(evidence, float)
    ids = np.asarray(event_ids)
    if z.ndim != 1 or y.shape != z.shape or ids.shape != z.shape:
        raise ValueError("Aligned one-dimensional observations required")
    if len(z) < 3 or not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("At least three finite observations required")
    if pd.isna(ids).any():
        raise ValueError("Event identity must be observed, not missing")
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    if len(counts) <= 2:
        return None
    x = np.column_stack((np.ones(len(z)), z))
    w = 1 / counts[inverse] if balance_events else np.ones(len(z))
    gram = x.T @ (w[:, None] * x)
    if np.linalg.matrix_rank(gram) < 2:
        return None
    bread = np.linalg.inv(gram)
    beta = bread @ (x.T @ (w * y))
    residual = y - x @ beta
    if cluster_errors:
        scores = np.zeros((len(counts), 2))
        np.add.at(scores, inverse, x * (w * residual)[:, None])
        correction = len(counts) / (len(counts) - 1) * (len(z) - 1) / (len(z) - 2)
        covariance = correction * bread @ (scores.T @ scores) @ bread
        dof = len(counts) - 2
    else:
        covariance = float(np.sum(w * residual**2) / (len(z) - 2)) * bread
        dof = len(z) - 2
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    radius = t.ppf(0.975, dof) * se
    cluster_mass = np.bincount(inverse, weights=w)
    return {
        "beta": beta,
        "covariance": covariance,
        "lower": beta - radius,
        "upper": beta + radius,
        "reports": len(z),
        "events": len(counts),
        "effective_events_perfect_dependence": float(
            w.sum() ** 2 / (cluster_mass @ cluster_mass)
        ),
    }


def resolve_revisions(frame):
    """Keep the latest explicit revision per author/event, preserving all history.

    Revisions must be directly linked and ordered by observed version, not by a
    guessed embedding similarity or input row order. Retractions remove an event
    from active evidence, not from the returned historical ledger.
    """
    required = {"author", "event", "version", "active", "score"}
    if not required <= set(frame):
        raise ValueError("Missing revision ledger fields")
    if frame[list(required)].isna().any().any():
        raise ValueError("Revision fields cannot be missing")
    if not pd.api.types.is_bool_dtype(frame.active):
        raise ValueError("Active flags must be boolean")
    if frame.duplicated(["author", "event", "version"]).any():
        raise ValueError("Ambiguous duplicate revision version")
    if not np.isfinite(frame.score).all() or not np.isfinite(frame.version).all():
        raise ValueError("Finite score and revision version required")
    history = frame.sort_values(["author", "event", "version"]).copy()
    latest = history.groupby(["author", "event"], sort=False).tail(1)
    return latest[latest.active].copy(), history


def calibration_fit(margins, signs):
    """Calibrate a fixed reader's signed score using source event labels only."""
    x, y = np.asarray(margins, float), np.asarray(signs, float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all():
        raise ValueError("Aligned finite source scores required")
    if set(np.unique(y)) != {-1.0, 1.0} or np.std(x) < 1e-10:
        raise ValueError("Both source directions and a varying reader required")
    design = np.column_stack((np.ones(len(x)), x))
    return np.linalg.lstsq(design, y, rcond=None)[0]
