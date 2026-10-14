#!/usr/bin/env python3
"""Finite author observation channels and qualified history evidence scoring."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_joint_evidence as J


def channel(p):
    """Return the conditional-expectation round trip, retaining missing cells."""
    p = np.asarray(p, float)
    if (
        p.ndim != 2
        or len(p) < 2
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1, rtol=0, atol=2e-12)
    ):
        raise ValueError("Finite normalized author observation rows required")
    r = p.mean(0)
    support = r > 0
    h = p[:, support] / np.sqrt(len(p) * r[support])
    centered = h - np.sqrt(r[support] / len(p))
    b = h @ h.T
    return {
        "reference": r,
        "support": support,
        "round_trip": b,
        "centered_round_trip": centered @ centered.T,
        "centered_spectrum": np.linalg.eigvalsh(centered.T @ centered)[::-1],
    }


def target_accounts(p, y):
    """Compute source label, posterior-field and return energies, not validity."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    c = channel(p)
    if y.ndim != 2 or y.shape[0] != len(p) or not np.isfinite(y).all():
        raise ValueError("Aligned finite author target functions required")
    g = y - y.mean(0)
    w = np.zeros((p.shape[1], y.shape[1]))
    s = c["support"]
    w[s] = p[:, s].T @ g / (len(p) * c["reference"][s, None])
    theta = p @ w
    return {
        "field": w,
        "return_scores": theta,
        "label_gram": g.T @ g / len(p),
        "field_gram": w.T @ (c["reference"][:, None] * w),
        "return_gram": theta.T @ theta / len(p),
        "literal_return_error": float(np.max(abs(theta - c["round_trip"] @ g))),
    }


def window_pair(p0, p1):
    """Select independently from two finite catalogues and expose coarsening."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if p0.shape != p1.shape or p0.ndim != 2:
        raise ValueError("Aligned two-window author catalogues required")
    channel(p0)
    channel(p1)
    m = p0.shape[1]
    q = (p0[:, :, None] * p1[:, None, :]).reshape(len(p0), m * m)
    garbling = np.zeros((m * m, m))
    for a, b in np.ndindex(m, m):
        garbling[m * a + b, a] += 0.5
        garbling[m * a + b, b] += 0.5
    return q, garbling


def cold_history(bundle, cells, event_ids):
    """Apply HM174's three laws to unique events, retaining support failures."""
    a = np.asarray(cells)
    if (
        a.ndim != 1
        or len(event_ids) != len(a)
        or any(not isinstance(i, str) for i in event_ids)
        or len(set(event_ids)) != len(a)
    ):
        raise ValueError("Distinct addressed events required;not repeated reports")
    result = J.evaluate(bundle, a)
    p = np.asarray(bundle["source_mass"])
    ll = J.log_likelihoods(p, J.counts(a))
    norm = logsumexp(ll)
    if np.isfinite(norm):
        weight = np.exp(ll - norm)
        effective = float(1 / (weight @ weight))
    else:
        weight = np.full(len(p), np.nan)
        effective = None
    return result | {
        "author_weights": weight,
        "effective_source_authors": effective,
    }


def history_groups(cells, masks, event_ids):
    """Preserve full/removal/subset event identities and all condition cells."""
    a, m = np.asarray(cells), np.asarray(masks)
    if (
        a.shape != (12,)
        or m.shape != (3, 12)
        or m.dtype != np.dtype(bool)
        or not m[0].all()
        or tuple(m.sum(1)) != (12, 8, 8)
        or np.any((~m[1]) & (~m[2]))
        or len(event_ids) != 12
        or len(set(event_ids)) != 12
    ):
        raise ValueError("Original full12/disjoint four-event removal masks required")
    all_masks = np.concatenate([m, (~m[1:3])], axis=0)
    return [
        (a[ix], [event_ids[i] for i in ix])
        for mask in all_masks
        for ix in [np.flatnonzero(mask)]
    ]
