#!/usr/bin/env python3
"""Observable response witnesses with condition composition retained explicitly."""

from __future__ import annotations

import numpy as np


def aligned_source(features, responses, groups):
    """Require finite aligned source vectors and complete response profiles."""
    x, y = np.asarray(features, float), np.asarray(responses, float)
    groups = np.asarray(groups)
    if (
        x.ndim != 2
        or y.ndim < 2
        or len(x) < 2
        or len(y) != len(x)
        or groups.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Aligned finite source and complete group addresses required")
    return x, y, groups


def response_components(features, responses, groups):
    """Compute total=within+between without removing conditional means."""
    x, y, g = aligned_source(features, responses, groups)
    origin, base = x.mean(0), y.mean(0)
    xc, yc = x - origin, y - base
    total = np.einsum("n...,nf->...f", yc, xc) / len(x)
    within, between = np.zeros_like(total), np.zeros_like(total)
    for key in np.unique(g):
        ids = g == key
        xg, yg = x[ids].mean(0), y[ids].mean(0)
        within += np.einsum("n...,nf->...f", y[ids] - yg, x[ids] - xg) / len(x)
        between += np.einsum("...,f->...f", yg - base, xg - origin) * (
            ids.sum() / len(x)
        )
    return origin, total, within, between


def relink_profiles(responses, groups, shift):
    """Rotate whole observed profiles within blocks, preserving conditional multisets."""
    y, g = np.asarray(responses, float), np.asarray(groups)
    if (
        y.ndim < 2
        or g.shape != (len(y),)
        or not np.isfinite(y).all()
        or not isinstance(shift, int)
        or shift < 1
    ):
        raise ValueError("Finite whole profiles and positive integer shift required")
    result = y.copy()
    for key in np.unique(g):
        ids = np.flatnonzero(g == key)
        if len(ids) <= shift:
            raise ValueError("Relink shift must be smaller than every source block")
        result[ids] = y[np.roll(ids, shift)]
    return result


def cold_readings(features, origin, coefficients):
    """Contract numeric article vectors with frozen response functions only."""
    x, m, c = (np.asarray(v, float) for v in (features, origin, coefficients))
    if (
        x.ndim != 2
        or m.shape != (x.shape[1],)
        or c.ndim < 2
        or c.shape[-1] != x.shape[1]
        or not all(np.isfinite(v).all() for v in (x, m, c))
    ):
        raise ValueError("Finite vectors and compatible frozen program required")
    return np.einsum("nf,...f->n...", x - m, c)


def order_fields(values):
    """Retain balanced response and order susceptibility in identical measurement units."""
    y = np.asarray(values, float)
    if y.ndim < 1 or y.shape[-1] != 2 or not np.isfinite(y).all():
        raise ValueError("Two finite semantically aligned option orders required")
    return np.stack(((y[..., 0] + y[..., 1]) / 2, (y[..., 0] - y[..., 1]) / 2), -1)


def signed_alignment(left, right):
    """Describe paired response agreement without fitting scale or converting to accuracy."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite([a, b]).all():
        raise ValueError("Nonempty aligned finite response differences required")
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return {
        "pairs": len(a),
        "cosine": float(a @ b / norm) if norm > 0 else None,
        "signed_product_mean": float(np.mean(a * b)),
        "same": int((a * b > 0).sum()),
        "opposite": int((a * b < 0).sum()),
        "zero": int((a * b == 0).sum()),
        "reading_mean": float(a.mean()),
        "response_mean": float(b.mean()),
        "reading_rms": float(np.sqrt(np.mean(a * a))),
        "response_rms": float(np.sqrt(np.mean(b * b))),
    }
