#!/usr/bin/env python3
"""Shared-query reference and centered-value form of within-comment attention."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.query_tilt import tilted_readout, validate_arrays


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


def directional_cumulants(keys, values, reference, native, coefficient):
    """Return exact finite-query response and its first two directional derivatives.

    For R=V@coefficient and U=K@(native-reference), derivatives of E[R]
    along the query segment are Cov(R,U) and E[(R-E[R])*(U-E[U])**2].
    These are local Taylor coefficients, not guaranteed finite-step approximations.
    """
    k, v, q0 = validate_arrays(keys, values, reference)
    _, _, q1 = validate_arrays(k, v, native)
    beta = np.asarray(coefficient, float)
    if beta.shape != (v.shape[1],) or not np.isfinite(beta).all():
        raise ValueError("Finite value-space readout coefficient required")
    r = v @ beta
    _, p = tilted_readout(k, v, q0)
    _, p1 = tilted_readout(k, v, q1)
    u = k @ (q1 - q0)
    rc, uc = r - p @ r, u - p @ u
    return {
        "shared": float(p @ r),
        "first": float(p @ (rc * uc)),
        "second": float(p @ (rc * uc**2)),
        "exact_increment": float((p1 - p) @ r),
        "directional_fisher": float(p @ uc**2),
    }


def pairing_query_expansion(cache, groups, reference, coefficient):
    """Sum frozen-head query derivatives without a decoder forward or backward."""
    q0, q1 = np.asarray(reference), np.asarray(cache["q"])
    k, v, beta = cache["k"], cache["v"], np.asarray(coefficient)
    if (
        q0.shape != q1.shape
        or q1.ndim != 2
        or len(q1) % k.shape[1]
        or beta.shape != (len(q1) * v.shape[-1],)
    ):
        raise ValueError("Compatible multi-head query, KV and reader shapes required")
    ratio, components = len(q1) // k.shape[1], []
    for head in range(len(q1)):
        kv = head // ratio
        start = head * v.shape[-1]
        components.append(
            directional_cumulants(
                k[:, kv],
                centered_values(v[:, kv], groups),
                q0[head],
                q1[head],
                beta[start : start + v.shape[-1]],
            )
        )
    return {key: float(sum(c[key] for c in components)) for key in components[0]}
