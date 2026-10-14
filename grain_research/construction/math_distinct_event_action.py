#!/usr/bin/env python3
"""Exact conditional pair actions; do not equate reweighting with removed evidence."""

import numpy as np

GEOMETRIES = ("squared", "distinct_event", "within_condition_term")
PIECES = ("choice_change", "survivor_normalization", "removed_pairs")
READINGS = ("level", *GEOMETRIES, *PIECES)


def pair_weights(conditions):
    """Return ordered unequal-event weights, or missing conditional opportunity."""
    c = np.asarray(conditions)
    if c.ndim != 1 or not np.isin(c, (0, 1)).all():
        raise ValueError("One binary condition per addressed event required")
    c = c.astype(int)
    n = np.bincount(c.astype(int), minlength=2)
    if (n < 2).any():
        return None
    same = c[:, None] == c[None, :]
    b = np.where(same, 1 / (n[c] * (n[c] - 1))[:, None], -1 / np.prod(n))
    np.fill_diagonal(b, 0)
    return b


def choice_weight(conditions):
    """Keep the fixed signed choice modulation, including balanced zero cases."""
    p = float(np.mean(conditions))
    return (1 - 2 * p) * p * (1 - p)


def geometries(vectors, conditions):
    """Read M/U/V on identical support; negative U is not clipped."""
    x, c = np.asarray(vectors, float), np.asarray(conditions)
    if x.ndim != 2 or len(x) != len(c) or not np.isfinite(x).all():
        raise ValueError("Finite event vectors and matched conditions required")
    b = pair_weights(c)
    if b is None:
        return None
    delta = x[c == 1].mean(0) - x[c == 0].mean(0)
    raw = np.outer(delta, delta)
    unequal = x.T @ b @ x
    return np.stack((raw, unequal, raw - unequal))


def deletion_action(vectors, conditions, removed, readers):
    """Separate removed pairs from surviving-pair normalization and choice action."""
    x, c, w = np.asarray(vectors, float), np.asarray(conditions), np.asarray(readers)
    d = np.asarray(removed)
    if d.ndim != 1 or not np.issubdtype(d.dtype, np.integer):
        raise ValueError("Integer event slots required")
    if len(np.unique(d)) != len(d) or ((d < 0) | (d >= len(x))).any():
        raise ValueError("Duplicate or unknown removed event")
    if w.ndim != 3 or w.shape[1:] != (x.shape[1], x.shape[1]):
        raise ValueError("One fixed square reader per source target required")
    if not np.isfinite(w).all() or not np.allclose(w, w.transpose(0, 2, 1)):
        raise ValueError("Finite symmetric readers required")
    keep = ~np.isin(np.arange(len(x)), d)
    old, new = geometries(x, c), geometries(x[keep], c[keep])
    if old is None or new is None:
        return None
    b0, b1 = pair_weights(c), pair_weights(c[keep])
    a0, a1 = choice_weight(c), choice_weight(c[keep])
    k = np.einsum("id,qde,je->qij", x, w, x)
    kk, bb = k[:, keep][:, :, keep], b0[keep][:, keep]
    choice = np.sum(kk * ((a1 - a0) * (b1 + bb) / 2), axis=(1, 2))
    survivor = np.sum(kk * ((a1 + a0) * (b1 - bb) / 2), axis=(1, 2))
    touches = ~(keep[:, None] & keep[None, :])
    removed_value = -np.sum(k * (a0 * b0 * touches), axis=(1, 2))
    old_scores = a0 * np.einsum("qij,hij->hq", w, old)
    new_scores = a1 * np.einsum("qij,hij->hq", w, new)
    return {
        "full": old_scores,
        "remaining": new_scores,
        "delta": new_scores - old_scores,
        "pieces": np.stack((choice, survivor, removed_value)),
    }


def author_mean(values, complete_views):
    """Average only complete SAME views; preserve unobserved authors as NaN."""
    x, s = np.asarray(values, float), np.asarray(complete_views, bool)
    if x.shape[:2] != s.shape:
        raise ValueError("Values belong to a different author-view panel")
    count = s.sum(1)
    expanded = s[(...,) + (None,) * (x.ndim - 2)]
    total = np.where(expanded, x, 0).sum(1)
    out = np.full_like(total, np.nan)
    den = count[(...,) + (None,) * (total.ndim - 1)]
    np.divide(total, den, out=out, where=den != 0)
    return out


def contact(x, y):
    """Describe centered contact and direction without fitting or sign rescue."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.isfinite(x) & np.isfinite(y)
    x, y = x[s], y[s]
    if len(x) < 2:
        return {
            "authors": len(x),
            "r": None,
            "covariance": None,
            "same_direction": 0,
            "opposite_direction": 0,
            "zero_direction": len(x),
        }
    dx, dy = x - x.mean(), y - y.mean()
    den = np.linalg.norm(dx) * np.linalg.norm(dy)
    return {
        "authors": len(x),
        "r": float(dx @ dy / den) if den else None,
        "covariance": float(dx @ dy / len(x)),
        "same_direction": int((x * y > 0).sum()),
        "opposite_direction": int((x * y < 0).sum()),
        "zero_direction": int((x * y == 0).sum()),
    }
