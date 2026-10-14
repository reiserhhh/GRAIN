#!/usr/bin/env python3
"""Separate author composition from attention partition mass, without fitting."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_cold_evidence import rotate


def registered_queries(bundle, endpoint):
    """Place every frozen source query in its declared record coordinates."""
    return np.stack(
        [
            rotate(raw, endpoint + gap, bundle["frequencies"], bundle["scaling"])
            for raw, gap in zip(bundle["raw_queries"], bundle["gaps"], strict=True)
        ]
    )


def common_key_offset(queries, key_heads, rcond=1e-12):
    """Compile a common logit shift from query geometry only, not observed scores."""
    q = np.asarray(queries, float)
    if q.ndim != 3 or q.shape[1] % key_heads or not np.isfinite(q).all():
        raise ValueError("Finite target/query-head/dimension geometry required")
    ratio = q.shape[1] // key_heads
    shifts, diagnostics = [], []
    for kh in range(key_heads):
        design = q[:, kh * ratio : (kh + 1) * ratio].reshape(-1, q.shape[2])
        shift, _, rank, singular = np.linalg.lstsq(
            design, np.ones(len(design)), rcond=rcond
        )
        shifts.append(shift)
        diagnostics.append(
            {
                "kv_head": kh,
                "rank": int(rank),
                "rows": len(design),
                "norm": float(np.linalg.norm(shift)),
                "condition": float(singular[0] / singular[-1]),
                "residual": float(np.max(np.abs(design @ shift - 1))),
            }
        )
    return np.stack(shifts), diagnostics


def observe(keys, values, queries, directions):
    """Return per-target/head normalized readings and token-sum log partitions."""
    k, v, q, d = (np.asarray(a, float) for a in (keys, values, queries, directions))
    if (
        k.ndim != 3
        or len(k) == 0
        or v.shape != k.shape
        or q.ndim != 3
        or d.shape != q.shape
        or q.shape[2] != k.shape[2]
        or q.shape[1] % k.shape[1]
        or not all(np.isfinite(a).all() for a in (k, v, q, d))
    ):
        raise ValueError("Finite compatible nonempty K/V/query/direction required")
    kh = np.arange(q.shape[1]) // (q.shape[1] // k.shape[1])
    logits = np.einsum("ahd,thd->ath", k[:, kh], q)
    content = np.einsum("ahd,thd->ath", v[:, kh], d)
    maximum = logits.max(0)
    mass = np.exp(logits - maximum)
    denominator = mass.sum(0)
    return (mass * content).sum(0) / denominator, maximum + np.log(denominator)


def compose(readings, log_mass, probabilities):
    """Return declared expectation, partition-weighted merge and exact gap."""
    f, z, p = (np.asarray(a, float) for a in (readings, log_mass, probabilities))
    if (
        f.ndim < 1
        or z.shape != f.shape
        or p.shape != (len(f),)
        or not all(np.isfinite(a).all() for a in (f, z, p))
        or np.min(p) < 0
        or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Aligned finite readings and a probability law required")
    shape = (-1,) + (1,) * (f.ndim - 1)
    p = p.reshape(shape)
    expected = (p * f).sum(0)
    scaled = np.exp(z - z.max(0))
    denom = (p * scaled).sum(0)
    merged = (p * scaled * f).sum(0) / denom
    covariance = (p * scaled * (f - expected)).sum(0) / denom
    return expected, merged, covariance


def merge_lifts(left, right):
    """Associatively add positive (log partition, normalized reading) lifts."""
    zl, fl = (np.asarray(x, float) for x in left)
    zr, fr = (np.asarray(x, float) for x in right)
    if not (zl.shape == fl.shape == zr.shape == fr.shape):
        raise ValueError("Aligned lift shapes required")
    if not all(np.isfinite(x).all() for x in (zl, fl, zr, fr)):
        raise ValueError("Positive finite lifts required; no fabricated empty score")
    z = np.logaddexp(zl, zr)
    return z, np.exp(zl - z) * fl + np.exp(zr - z) * fr


def author_laws():
    """Positive three-state cycle laws with identical marginal traffic."""
    d = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], float)
    return ((1 + 0.5 * d) / 9).ravel(), ((1 - 0.5 * d) / 9).ravel()


def cold_score(reader, carriers):
    """Read numeric K/V once; no text, target answer, model or design truth input."""
    keys, values, offsets, endpoints = (
        carriers[n] for n in ("keys", "values", "offsets", "endpoints")
    )
    if (
        len(offsets) != len(endpoints) + 1
        or offsets[0] != 0
        or offsets[-1] != len(keys)
        or (np.diff(offsets) <= 0).any()
    ):
        raise ValueError("Nonempty exact record boundaries required")
    f, z = [], []
    for start, end, endpoint in zip(offsets[:-1], offsets[1:], endpoints, strict=True):
        q = np.asarray(reader["positions"][str(int(endpoint))]["queries"])
        value, mass = observe(
            keys[start:end], values[start:end], q, reader["directions"]
        )
        f.append(value)
        z.append(mass)
    return np.stack(f), np.stack(z)
