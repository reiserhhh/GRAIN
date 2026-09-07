#!/usr/bin/env python3
"""Finite exponential-family query readouts and key/value pairing identities."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax


def validate_arrays(keys, values, query):
    """Validate paired finite key/value rows and an effective scaled query."""
    k, v, q = (
        np.asarray(keys, float),
        np.asarray(values, float),
        np.asarray(query, float),
    )
    if (
        k.ndim != 2
        or v.ndim != 2
        or len(k) != len(v)
        or not len(k)
        or q.shape != (k.shape[1],)
        or not all(np.isfinite(x).all() for x in (k, v, q))
    ):
        raise ValueError("Require paired finite keys/values and compatible query")
    return k, v, q


def tilted_readout(keys, values, query):
    """Compute E_q[V] and finite exponential weights; scale belongs in query."""
    k, v, q = validate_arrays(keys, values, query)
    p = softmax(k @ q)
    return p @ v, p


def query_jacobian(keys, values, query):
    """Return d E_q[V]/dq = Cov_q(V,K), not necessarily a symmetric matrix."""
    k, v, q = validate_arrays(keys, values, query)
    mean_v, p = tilted_readout(k, v, q)
    return ((v - mean_v) * p[:, None]).T @ (k - p @ k)


def integrated_query_response(keys, values, start, end, order=16):
    """Integrate the covariance action along a fixed-KV query segment numerically."""
    k, v, end = validate_arrays(keys, values, end)
    _, _, start = validate_arrays(k, v, start)
    if not isinstance(order, int) or order < 1:
        raise ValueError("Quadrature order must be a positive integer")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    delta = end - start
    projected_keys = k @ delta
    result = np.zeros(v.shape[1])
    for t, weight in zip((nodes + 1) / 2, weights / 2, strict=True):
        mean_v, p = tilted_readout(k, v, start + t * delta)
        result += weight * ((p * (projected_keys - p @ projected_keys)) @ (v - mean_v))
    return result


def pairing_component(weights, values, groups):
    """Return within-group KV-pairing contribution, preserving other value rows.

    For each selected group G this is sum_i (p_i-P_G/|G|) v_i. Subtracting
    the sum from the head output exactly averages over independent uniform
    permutations of values within groups, with attention weights fixed.
    """
    p, v = np.asarray(weights, float), np.asarray(values, float)
    if (
        v.ndim != 2
        or p.shape != (len(v),)
        or not np.isfinite(p).all()
        or not np.isfinite(v).all()
        or np.any(p < 0)
        or not np.isclose(p.sum(), 1)
    ):
        raise ValueError("Require finite values and probability weights")
    used, total = set(), np.zeros(v.shape[1])
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
            raise ValueError("Require nonempty disjoint valid token groups")
        used.update(ix)
        total += (p[ix] - p[ix].mean()) @ v[ix]
    return total


def query_covariance_split(keys, values, query, group_labels):
    """Split a fixed-KV query Jacobian by the law of total cross-covariance."""
    k, v, q = validate_arrays(keys, values, query)
    labels = np.asarray(group_labels)
    if labels.shape != (len(k),):
        raise ValueError("One declared group label per token is required")
    mean_v, p = tilted_readout(k, v, q)
    mean_k = p @ k
    within, between = (
        np.zeros((v.shape[1], k.shape[1])),
        np.zeros((v.shape[1], k.shape[1])),
    )
    for label in np.unique(labels):
        ix = np.flatnonzero(labels == label)
        mass = p[ix].sum()
        if mass == 0:
            continue
        pg = p[ix] / mass
        vg, kg = pg @ v[ix], pg @ k[ix]
        within += mass * (((v[ix] - vg) * pg[:, None]).T @ (k[ix] - kg))
        between += mass * np.outer(vg - mean_v, kg - mean_k)
    return within, between
