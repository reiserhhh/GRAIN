#!/usr/bin/env python3
"""Positive source-vector observation laws and their target score fields."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

VERSION = "HM204_source_vector_likelihood_v1"
TARGETS = ("E", "N", "A", "C", "O")


def validate(prototypes, labels, variance):
    """Require finite prototypes, five binary anchors and both sides per target."""
    z, y = np.asarray(prototypes, float), np.asarray(labels)
    if (
        z.ndim != 2
        or len(z) < 2
        or z.shape[1] == 0
        or not np.isfinite(z).all()
        or y.shape != (len(z), 5)
        or not np.isin(y, (0, 1)).all()
        or not np.all((y.mean(0) > 0) & (y.mean(0) < 1))
        or not np.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError(
            "Finite source law with positive variance and binary anchors required"
        )
    return z, y.astype(float), float(variance)


def source_variance(prototypes):
    """Use one label-free median pair-distance scale, with no search or fallback."""
    z = np.asarray(prototypes, float)
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("Finite source prototypes required")
    distances = np.maximum(
        np.sum(z * z, 1)[:, None] + np.sum(z * z, 1)[None, :] - 2 * z @ z.T,
        0,
    )
    d = distances[np.triu_indices(len(z), 1)]
    d = d[d > 0]
    if not len(d):
        raise ValueError("Coincident prototypes do not define a source variance")
    return float(np.median(d))


def read(prototypes, labels, variance, points):
    """Compute a shared joint-profile posterior and all target gradient fields."""
    z, y, s2 = validate(prototypes, labels, variance)
    x = np.asarray(points, float)
    if x.ndim != 2 or x.shape[1] != z.shape[1] or not np.isfinite(x).all():
        raise ValueError("Finite points in the declared vector coordinates required")
    d2 = np.maximum(
        np.sum(x * x, 1)[:, None] + np.sum(z * z, 1)[None, :] - 2 * x @ z.T,
        0,
    )
    log_kernel = -d2 / (2 * s2)
    partition = logsumexp(log_kernel, axis=1)
    w = np.exp(log_kernel - partition[:, None])
    mu = w @ y
    profile = (y @ (2 ** np.arange(5))).astype(int)
    full_profile = w @ np.eye(32)[profile]
    odds, fields = [], []
    for q in range(5):
        high = y[:, q] == 1
        zh, zl = logsumexp(log_kernel[:, high], 1), logsumexp(log_kernel[:, ~high], 1)
        wh = np.exp(log_kernel[:, high] - zh[:, None])
        wl = np.exp(log_kernel[:, ~high] - zl[:, None])
        odds.append(zh - zl)
        fields.append((wh @ z[high] - wl @ z[~high]) / s2)
    field = np.stack(fields, axis=1)
    odds = np.stack(odds, axis=1)
    return {
        "posterior": mu,
        "relative": mu - y.mean(0),
        "profile": full_profile,
        "log_odds": odds,
        "log_density": partition
        - np.log(len(z))
        - z.shape[1] / 2 * np.log(2 * np.pi * s2),
        "shared_density_score": (w @ z - x) / s2,
        "target_log_odds_field": field,
        "target_probability_field": field * (mu * (1 - mu))[:, :, None],
        "effective_prototypes": 1 / np.sum(w * w, axis=1),
        "minimum_distance": np.sqrt(d2.min(1)),
        "sigmoid_error": float(np.max(np.abs(expit(odds) - mu))) if len(x) else 0.0,
    }


def literal(prototypes, labels, variance, point):
    """Independent scalar-distance Gaussian mixture, not a Gram-distance replay."""
    z, y, s2 = validate(prototypes, labels, variance)
    x = np.asarray(point, float)
    d = np.array(
        [sum(float(a - b) ** 2 for a, b in zip(x, row, strict=True)) for row in z]
    )
    k = np.exp(-(d - d.min()) / (2 * s2))
    w = k / k.sum()
    mu = np.array([sum(w[i] * y[i, q] for i in range(len(z))) for q in range(5)])
    return mu


def cold_point(record, dimensions):
    """Bind a subset mean to identified evidence without reading text or labels."""
    if set(record) != {"frame_key", "events"} or not isinstance(
        record["frame_key"], str
    ):
        raise ValueError("Only addressed vector observations allowed")
    events = record["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("An observed event set is required")
    seen, vectors = set(), []
    for event in events:
        if set(event) != {"event_id", "vector"} or not isinstance(
            event["event_id"], str
        ):
            raise ValueError("Only identified vectors allowed")
        if not event["event_id"] or event["event_id"] in seen:
            raise ValueError("Repeated evidence identities are not new observations")
        seen.add(event["event_id"])
        vector = np.asarray(event["vector"], float)
        if vector.shape != (dimensions,) or not np.isfinite(vector).all():
            raise ValueError("Vector coordinates changed")
        vectors.append(vector)
    return np.mean(vectors, axis=0)
