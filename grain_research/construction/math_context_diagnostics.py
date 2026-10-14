#!/usr/bin/env python3
"""Relative diagnostic fields on a finite vocabulary, without predictor fitting."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def aligned_states(states, partitions):
    """Align carried/reset second slices; reject missing native observations."""
    h, a = np.asarray(states, float), np.asarray(partitions, float)
    if h.shape != (4, 32, 1536) or a.shape != (4, 32):
        raise ValueError("Full HM152 native observation shape required")
    if not np.isfinite(h).all() or not np.isfinite(a).all():
        raise ValueError("Nonfinite observation")
    hc, lc, hr, lr = h[0, 16:], h[1, 16:], h[2, :16], h[3, :16]
    delta = (hc - lc) - (hr - lr)
    midpoint = (hr + lr) / 2
    normalizer = a[0, 16:] - a[1, 16:] - a[2, :16] + a[3, :16]
    return delta, midpoint, normalizer


def field_account(field, reference_logits, normalizer, token_ids):
    """Separate q-centered diagnostic shape from constant calibration offsets."""
    u, z = np.asarray(field, float), np.asarray(reference_logits, float)
    a, ids = np.asarray(normalizer, float), np.asarray(token_ids)
    if u.ndim != 2 or z.shape != u.shape or a.shape != u.shape[:1]:
        raise ValueError("Aligned full-vocabulary fields required")
    if ids.shape != a.shape or ids.dtype.kind not in "iu":
        raise ValueError("Native token identifiers required")
    if ids.min() < 0 or ids.max() >= u.shape[1]:
        raise ValueError("Token outside vocabulary")
    if not all(np.isfinite(v).all() for v in (u, z, a)):
        raise ValueError("Nonfinite field")
    q = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    mean = np.sum(q * u, axis=1)
    centered = u - mean[:, None]
    selected = u[np.arange(len(ids)), ids]
    lo, hi = u.argmin(1), u.argmax(1)
    return {
        "selected_u": selected,
        "reference_mean_u": mean,
        "selected_shape": selected - mean,
        "calibration": mean - a,
        "reconstructed_k": selected - a,
        "weighted_std": np.sqrt(np.sum(q * centered**2, axis=1)),
        "oscillation": u.max(1) - u.min(1),
        "min_id": lo,
        "max_id": hi,
        "min_u": u[np.arange(len(ids)), lo],
        "max_u": u[np.arange(len(ids)), hi],
        "min_reference_probability": q[np.arange(len(ids)), lo],
        "max_reference_probability": q[np.arange(len(ids)), hi],
        "reference_zero_count": (q == 0).sum(1),
    }


def cross_fields(first, second, reference_logits):
    """Compare two target fields on exactly the same text and reference law."""
    e, n, z = [np.asarray(v, float) for v in (first, second, reference_logits)]
    if e.ndim != 2 or n.shape != e.shape or z.shape != e.shape:
        raise ValueError("Same support required")
    if not all(np.isfinite(v).all() for v in (e, n, z)):
        raise ValueError("Nonfinite target field")
    q = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    ec = e - np.sum(q * e, 1)[:, None]
    nc = n - np.sum(q * n, 1)[:, None]
    ve, vn = np.sum(q * ec**2, 1), np.sum(q * nc**2, 1)
    covariance = np.sum(q * ec * nc, 1)
    denominator = np.sqrt(ve * vn)
    cosine = np.divide(
        covariance, denominator, out=np.full(len(e), np.nan), where=denominator > 0
    )
    shared, specific = (ec + nc) / 2, (ec - nc) / 2
    return {
        "variance_E": ve,
        "variance_N": vn,
        "covariance": covariance,
        "cosine": cosine,
        "shared_variance": np.sum(q * shared**2, 1),
        "specific_variance": np.sum(q * specific**2, 1),
    }
