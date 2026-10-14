#!/usr/bin/env python3
"""Positive pair-law transport and its target-observer defect.

Established finite Markov/exterior algebra, not a new personality theorem.
No application target values enter the interpolation or transported law.
"""

from __future__ import annotations

import itertools

import numpy as np


def probability_map(reference, application, match_atol=1e-12):
    """Use fixed inverse-squared-distance weights with source cardinality.

    Distinct reference points are required. Exact matches give one-hot rows;
    other points receive a positive partition of unity. This is nonlinear in
    point coordinates, bounded and defined outside their affine hull.
    """
    x, y = np.asarray(reference, float), np.asarray(application, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or len(x) < 3:
        raise ValueError("Aligned reference and application points required")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or match_atol <= 0:
        raise ValueError("Finite coordinates and a positive match tolerance required")
    d0 = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    if np.any(d0[np.triu_indices(len(x), 1)] <= match_atol):
        raise ValueError("Reference points must be distinguishable")
    d = np.linalg.norm(y[:, None] - x[None, :], axis=-1)
    t = np.zeros_like(d)
    for i, distance in enumerate(d):
        matched = np.flatnonzero(distance <= match_atol)
        if len(matched) > 1:
            raise ValueError("Ambiguous exact match")
        if len(matched):
            t[i, matched[0]] = 1
        else:
            weights = (distance.min() / distance) ** 2
            t[i] = weights / weights.sum()
    return t


def checked_map(mapping):
    """Validate nonnegative row-stochastic coordinates without normalizing them."""
    t = np.asarray(mapping, float)
    if t.ndim != 2 or min(t.shape) < 3 or not np.isfinite(t).all():
        raise ValueError("Finite rectangular state map required")
    if t.min() < 0 or not np.allclose(t.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("A probability map must be nonnegative and row stochastic")
    return t


def pullback(kernel, mapping):
    """Transport every source relation channel to application state pairs."""
    t, f = checked_map(mapping), np.asarray(kernel, float)
    if f.ndim != 3 or f.shape[:2] != (t.shape[1], t.shape[1]):
        raise ValueError("Aligned source pair kernel with channels required")
    if not np.isfinite(f).all():
        raise ValueError("Finite source kernel required")
    return np.einsum("ai,ijq,bj->abq", t, f, t)


def pushforward(law, mapping):
    """Push a stationary law OR signed current; do not erase either object."""
    t, p = checked_map(mapping), np.asarray(law, float)
    if p.ndim != 2 or p.shape != (len(t), len(t)) or not np.isfinite(p).all():
        raise ValueError("Aligned finite application pair array required")
    return t.T @ p @ t


def fundamental_currents(n):
    """Return all source-independent triangular cycles and positive-law pairs."""
    if n < 3:
        raise ValueError("At least three nodes required")
    currents, triples = [], []
    for i, j in itertools.combinations(range(1, n), 2):
        d = np.zeros((n, n))
        for a, b in ((0, i), (i, j), (j, 0)):
            d[a, b], d[b, a] = 1, -1
        currents.append(d)
        triples.append((0, i, j))
    d = np.array(currents)
    base, scale = np.full((n, n), 1 / n**2), 1 / (2 * n**2)
    return d, base[None] + scale * d, base[None] - scale * d, triples


def centered_map_spectrum(mapping):
    """Return zero-sum node singular values and all exterior-square products.

    These are geometric amplitude diagnostics, NOT psychological reliability
    or a target-selected dimensionality rule. The maps may be rectangular.
    """
    t = checked_map(mapping)
    a, b = t.shape
    h0, h1 = np.eye(a) - 1 / a, np.eye(b) - 1 / b
    centered = h0 @ t @ h1
    singular = np.linalg.svd(centered, compute_uv=False)
    exterior = np.array([x * y for x, y in itertools.combinations(singular, 2)])
    return centered, singular, exterior
