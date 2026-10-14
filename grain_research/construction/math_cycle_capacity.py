#!/usr/bin/env python3
"""Finite target-relative observation capacity on a complete evidence graph.

These are established exterior-algebra/Hodge/projection constructions, not a
new personality theorem. The metric and finite state support are explicit.
"""

from __future__ import annotations

import itertools

import numpy as np


def cycles(kernel):
    """Remove exchange-even and exact-gradient parts on a complete graph."""
    f = np.asarray(kernel, float)
    if f.ndim < 2 or f.shape[0] != f.shape[1] or len(f) < 3:
        raise ValueError("At least three aligned states required")
    if not np.isfinite(f).all():
        raise ValueError("Finite kernels required")
    odd = (f - f.swapaxes(0, 1)) / 2
    potential = odd.sum(axis=1) / len(f)
    gradient = potential[:, None] - potential[None, :]
    return odd, gradient, odd - gradient


def area_bank(points, anchor):
    """Return every coordinate-plane triangular area, with no selected axis."""
    x, z = np.asarray(points, float), np.asarray(anchor, float)
    if x.ndim != 2 or len(x) < 3 or z.shape != (x.shape[1],):
        raise ValueError("Aligned node coordinates and one anchor required")
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("Finite coordinates required")
    v = x - z
    pairs = list(itertools.combinations(range(x.shape[1]), 2))
    if not pairs:
        return np.empty((len(x), len(x), 0))
    return np.stack(
        [
            (np.outer(v[:, a], v[:, b]) - np.outer(v[:, b], v[:, a])) / 2
            for a, b in pairs
        ],
        axis=-1,
    )


def edge_columns(kernel):
    """Use one orientation per edge; Frobenius inner products are twice this."""
    f = np.asarray(kernel, float)
    i, j = np.triu_indices(len(f), 1)
    return f[i, j].reshape(len(i), -1)


def orthogonal_basis(columns, rtol=1e-10, atol=1e-12):
    """Declare numerical rank using one frozen cutoff, not an accuracy search."""
    x = np.asarray(columns, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite column matrix required")
    if x.shape[1] == 0:
        return np.empty((len(x), 0)), np.empty(0), atol
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    cutoff = max(atol, rtol * s[0]) if len(s) else atol
    return u[:, s > cutoff], s, cutoff


def decompose(bank, targets, rtol=1e-10, atol=1e-12):
    """Split measured target cochains into observed and invisible components.

    This is a post-observation support decomposition, NOT a trained predictor
    or independent transport test. No target scores choose the bank or rank.
    """
    _, _, a = cycles(bank)
    _, _, f = cycles(targets)
    aedges, fedges = edge_columns(a), edge_columns(f)
    basis, singular, cutoff = orthogonal_basis(aedges, rtol, atol)
    shared_edges = basis @ (basis.T @ fedges)
    residual_edges = fedges - shared_edges
    shared, residual = np.zeros_like(f), np.zeros_like(f)
    i, j = np.triu_indices(len(f), 1)
    shared[i, j], shared[j, i] = shared_edges, -shared_edges
    residual[i, j], residual[j, i] = residual_edges, -residual_edges
    coefficients = np.linalg.lstsq(aedges, shared_edges, rcond=rtol)[0]
    return {
        "area_cycle": a,
        "target_cycle": f,
        "shared": shared,
        "residual": residual,
        "basis": basis,
        "singular": singular,
        "cutoff": cutoff,
        "coefficients": coefficients,
    }


def legal_witness(residual, atol=1e-10):
    """Construct positive stationary pair laws invisible to the chosen bank.

    The witness is derived from the measured residual: an existence example,
    not independently sampled author evidence or a held-out prediction.
    """
    r = np.asarray(residual, float)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("One square residual channel required")
    if not np.isfinite(r).all() or not np.allclose(r, -r.T, atol=atol, rtol=0):
        raise ValueError("Finite skew residual required")
    if np.max(np.abs(r.sum(1))) > atol:
        raise ValueError("Witness must be divergence-free")
    n, size = len(r), np.max(np.abs(r))
    base = np.full((n, n), 1 / n**2)
    scale = 1 / (2 * n**2 * size) if size > atol else 0.0
    return base + scale * r, base - scale * r, scale
