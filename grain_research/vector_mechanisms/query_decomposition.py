#!/usr/bin/env python3
"""Shared-query reference and centered-value form of within-comment attention."""

from __future__ import annotations

import numpy as np


def centered_values(values, groups):
    """Represent group pairing as a single softmax expectation, including mass changes."""
    v = np.asarray(values, float)
    if v.ndim != 2 or not np.isfinite(v).all():
        raise ValueError("Finite token/value matrix required")
    out, used = np.zeros_like(v), set()
    for group in groups:
        ix = np.asarray(group)
        if (
            ix.ndim != 1
            or not len(ix)
            or not np.issubdtype(ix.dtype, np.integer)
            or ix.min() < 0
            or ix.max() >= len(v)
            or len(set(ix)) != len(ix)
            or used.intersection(ix)
        ):
            raise ValueError("Require disjoint nonempty token groups")
        used.update(ix)
        out[ix] = v[ix] - v[ix].mean(0)
    return out


def fold_query_references(raw_queries, folds):
    """Compute equal-author/equal-view reference queries excluding the held-out fold."""
    q, folds = np.asarray(raw_queries, float), np.asarray(folds)
    if q.ndim != 4 or folds.shape != (len(q),) or not np.isfinite(q).all():
        raise ValueError("Require author/view/head/coordinate array and matched folds")
    values = np.unique(folds)
    if len(values) < 2:
        raise ValueError("At least two folds required")
    mean = q.mean(axis=1)
    return {int(fold): mean[folds != fold].mean(axis=0) for fold in values}


def mix_raw_query(reference, native, alpha):
    """Interpolate declared raw query endpoints, preserving native endpoint exactly."""
    a, b = np.asarray(reference), np.asarray(native)
    if a.shape != b.shape or not np.isfinite([a, b]).all() or not 0 <= alpha <= 1:
        raise ValueError(
            "Paired finite reference/native queries and bounded dose required"
        )
    if alpha == 1:
        return b.copy()
    if alpha == 0:
        return a.copy()
    return a + alpha * (b - a)
