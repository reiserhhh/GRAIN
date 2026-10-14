#!/usr/bin/env python3
"""Exact observable history bridges, not unique neural causal attribution."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import softmax

from grain_research.construction.math_cold_evidence import rotate

FACTORS = ("value", "registered_key", "subject_gate")
ORDERS = tuple(itertools.permutations(range(3)))


def primitives(source, exported, index):
    """Read source-defined covectors on query-independent current-token arrays."""
    lo, hi = exported["offsets"][index : index + 2]
    k, v = exported["keys"][lo:hi], exported["values"][lo:hi]
    gate = exported["gates"][lo:hi].T
    heads = source["directions"].shape[1]
    if heads % k.shape[1] or v.shape != k.shape or gate.shape != (heads, hi - lo):
        raise ValueError("Compatible GQA and body gates required")
    values = np.repeat(v, heads // v.shape[1], axis=1)
    l, c = [], []
    for qi, direction in enumerate(source["directions"]):
        q = rotate(
            source["raw_queries"][qi],
            int(exported["endpoints"][index]) + int(source["gaps"][qi]),
            source["frequencies"],
            float(source["scaling"]),
        )
        l.append(
            np.stack([k[:, h // (heads // k.shape[1])] @ q[h] for h in range(heads)])
        )
        c.append(np.einsum("ihd,hd->hi", values, direction))
    l, c = np.asarray(l), np.asarray(c)
    g = np.broadcast_to(gate, l.shape)
    return (
        np.stack((np.zeros_like(l), l, l)),
        np.stack((c, c, c)),
        np.stack((np.ones_like(g), np.ones_like(g), g)),
    )


def checked(logits, values, gates):
    """Require finite aligned target/head/token arrays and probability gates."""
    l, c, g = (np.asarray(a, float) for a in (logits, values, gates))
    if l.ndim < 2 or c.shape != l.shape or g.shape != l.shape or l.shape[-1] == 0:
        raise ValueError("Nonempty aligned head/token arrays required")
    if not all(np.isfinite(a).all() for a in (l, c, g)):
        raise ValueError("Finite observable primitives required")
    if np.any(g < 0) or np.any(g > 1):
        raise ValueError("Subject gates must be probabilities")
    return l, c, g


def score(logits, values, gates):
    """Use the inherited unnormalized gate numerator, summing every head."""
    l, c, g = checked(logits, values, gates)
    return np.sum(softmax(l, axis=-1) * c * g, axis=(-2, -1))


def cube(low, high):
    """Return all eight fixed primitive swaps, not a selected intervention."""
    l0, c0, g0 = checked(*low)
    l1, c1, g1 = checked(*high)
    if l0.shape != l1.shape:
        raise ValueError("Same current-token support required")
    return np.stack(
        [
            score(
                l1 if mask & 2 else l0, c1 if mask & 1 else c0, g1 if mask & 4 else g0
            )
            for mask in range(8)
        ],
        axis=-1,
    )


def finite_accounts(corners):
    """Keep six update orders, exact Shapley mean and all interactions."""
    y = np.asarray(corners, float)
    if y.shape[-1] != 8 or not np.isfinite(y).all():
        raise ValueError("Eight finite observable corners required")
    orders = np.zeros(y.shape[:-1] + (6, 3))
    for p, order in enumerate(ORDERS):
        mask = 0
        for factor in order:
            next_mask = mask | (1 << factor)
            orders[..., p, factor] = y[..., next_mask] - y[..., mask]
            mask = next_mask
    interactions = np.zeros_like(y)
    for mask in range(8):
        sub = mask
        while True:
            sign = (-1) ** ((mask ^ sub).bit_count())
            interactions[..., mask] += sign * y[..., sub]
            if sub == 0:
                break
            sub = (sub - 1) & mask
    return {"orders": orders, "shapley": orders.mean(-2), "interactions": interactions}


def derivative_terms(low, high, position):
    """Evaluate value, gate and logit covariance derivatives on a straight path."""
    l0, c0, g0 = checked(*low)
    l1, c1, g1 = checked(*high)
    if l0.shape != l1.shape or not 0 <= position <= 1:
        raise ValueError("Aligned endpoints and path position in [0,1] required")
    dl, dc, dg = l1 - l0, c1 - c0, g1 - g0
    l, c, g = l0 + position * dl, c0 + position * dc, g0 + position * dg
    p = softmax(l, axis=-1)
    mean_dl = np.sum(p * dl, axis=-1, keepdims=True)
    value = np.sum(p * g * dc, axis=(-2, -1))
    gate = np.sum(p * dg * c, axis=(-2, -1))
    key = np.sum(p * g * c * (dl - mean_dl), axis=(-2, -1))
    return np.stack((value, key, gate), axis=-1)


def integral_accounts(low, high, nodes=16):
    """Integrate the declared path; quadrature does not select a predictor."""
    if not isinstance(nodes, int) or nodes < 2:
        raise ValueError("At least two fixed quadrature nodes required")
    positions, weights = np.polynomial.legendre.leggauss(nodes)
    terms = np.stack(
        [derivative_terms(low, high, float((s + 1) / 2)) for s in positions],
        axis=0,
    )
    return np.tensordot(weights / 2, terms, axes=(0, 0))


def current_statistics(component, total):
    """Signed alignment and norm ratios, not variance/psychological fractions."""
    a, t = np.asarray(component, float), np.asarray(total, float)
    if a.shape != t.shape or a.ndim != 3 or not np.isfinite([a, t]).all():
        raise ValueError("Aligned finite cycle matrices and channels required")
    energy = np.sum(t * t, axis=(0, 1))
    inner = np.sum(a * t, axis=(0, 1))
    own = np.sum(a * a, axis=(0, 1))
    return {
        "rms": np.sqrt(own / (a.shape[0] * a.shape[1])),
        "signed_alignment": np.divide(
            inner, energy, out=np.full_like(inner, np.nan), where=energy > 0
        ),
        "norm_ratio": np.sqrt(
            np.divide(own, energy, out=np.full_like(own, np.nan), where=energy > 0)
        ),
    }
