#!/usr/bin/env python3
"""Exact gain/allocation decomposition and finite conditional cumulant readouts."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, xlogy

from grain_research.vector_mechanisms.query_decomposition import centered_values
from grain_research.vector_mechanisms.query_tilt import validate_arrays

ARMS = (
    "exact",
    "gain_only_exact",
    "exact_mass_cov",
    "gain_only_cov",
    "exact_mass_second",
    "uniform_cov",
)


def gain_readouts(keys, values, query, groups):
    """Separate total body gain, comment allocation and conditional readout shape.

    The normalizer includes the full captured support. Conditional second order
    is a Taylor approximation about uniform weights inside each comment, with
    native comment masses retained. No fitted calibration is performed.
    """
    k, v, q = validate_arrays(keys, values, query)
    centered = centered_values(v, groups)
    if not len(groups):
        raise ValueError("At least one nonempty comment group required")
    logits = k @ q
    log_total = logsumexp(logits)
    group_logs, exact, linear, second, counts = [], [], [], [], []
    for group in groups:
        ix = np.asarray(group)
        l, r = logits[ix], centered[ix]
        z = logsumexp(l)
        lc = l - l.mean()
        group_logs.append(z)
        counts.append(len(ix))
        exact.append(np.exp(l - z) @ r)
        linear.append(lc @ r / len(ix))
        second.append(lc**2 @ r / len(ix))
    group_logs = np.asarray(group_logs)
    counts = np.asarray(counts)
    log_body = logsumexp(group_logs)
    body_gain = float(np.exp(log_body - log_total))
    mass = np.exp(group_logs - log_total)
    allocation = np.exp(group_logs - log_body)
    proportions = counts / counts.sum()
    exact, linear, second = map(np.asarray, (exact, linear, second))
    vectors = {
        "exact": mass @ exact,
        "gain_only_exact": body_gain * (proportions @ exact),
        "exact_mass_cov": mass @ linear,
        "gain_only_cov": body_gain * (proportions @ linear),
        "exact_mass_second": mass @ (linear + 0.5 * second),
        "uniform_cov": (counts / len(k)) @ linear,
    }
    entropy = float(-np.sum(xlogy(allocation, allocation)))
    return vectors, {
        "body_mass": body_gain,
        "outside_mass": float(1 - body_gain),
        "first_token_mass": float(np.exp(logits[0] - log_total)),
        "comment_entropy": entropy,
        "normalized_comment_entropy": entropy / np.log(len(groups))
        if len(groups) > 1
        else 0.0,
        "body_token_fraction": float(counts.sum() / len(k)),
        "group_mass": mass.tolist(),
    }


def multihead_gain(cache, groups, query, coefficient):
    """Apply each head's gain decomposition and preserve the affine reader basis."""
    k, v, q, beta = cache["k"], cache["v"], np.asarray(query), np.asarray(coefficient)
    if q.ndim != 2 or len(q) % k.shape[1] or beta.shape != (len(q) * v.shape[-1],):
        raise ValueError("Compatible head, KV and coefficient dimensions required")
    ratio, parts, diagnostics = len(q) // k.shape[1], {arm: [] for arm in ARMS}, []
    for head, query_head in enumerate(q):
        vectors, detail = gain_readouts(
            k[:, head // ratio], v[:, head // ratio], query_head, groups
        )
        block = beta[head * v.shape[-1] : (head + 1) * v.shape[-1]]
        detail["head"] = head
        detail["contributions"] = {
            arm: float(vector @ block) for arm, vector in vectors.items()
        }
        diagnostics.append(detail)
        for arm in ARMS:
            parts[arm].append(vectors[arm])
    return {arm: np.concatenate(values) for arm, values in parts.items()}, diagnostics
