#!/usr/bin/env python3
"""Identity-aware source moments, distinct from mean-atlas repeatability."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_joint_target_atlas import characters


def kernel_spaces(means, basis):
    """Keep full, retained and complementary cross-view kernels at two origins."""
    x, q = np.asarray(means, float), np.asarray(basis, float)
    if (
        x.ndim != 3
        or x.shape[1] != 2
        or not np.isfinite(x).all()
        or q.ndim != 2
        or q.shape[0] != x.shape[-1]
        or not np.isfinite(q).all()
        or not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=2e-12, rtol=0)
    ):
        raise ValueError("Finite paired views and fixed orthonormal basis required")
    kernels = []
    for z in (x, x - x.mean(0, keepdims=True)):
        retained = z @ q
        full = z[:, 0] @ z[:, 1].T
        projected = retained[:, 0] @ retained[:, 1].T
        kernels.append(np.array((full, projected, full - projected)))
    return np.asarray(kernels)


def accounts(kernel, codes):
    """Resolve all character moments by actual author identity, with no fit."""
    k, c = np.asarray(kernel, float), np.asarray(codes)
    if (
        c.ndim != 1
        or c.dtype.kind not in "iu"
        or not np.isin(c, np.arange(32)).all()
        or k.shape != (len(c), len(c))
        or not np.isfinite(k).all()
    ):
        raise ValueError("One finite source kernel and32-profile addresses required")
    n = np.bincount(c, minlength=32)
    if n.min() == 0:
        raise ValueError("Missing profile cannot be reweighted or set to zero")
    h = characters()
    pair_codes = 32 * c[:, None] + c[None, :]
    totals = np.bincount(pair_codes.ravel(), weights=k.ravel(), minlength=1024).reshape(
        32, 32
    )
    means = totals / (n[:, None] * n[None, :])
    same_author = np.bincount(c, weights=np.diag(k), minlength=32)
    diagonal = h.T @ ((same_author / n**2)[:, None] * h) / 1024
    v = h.T @ means @ h / 1024
    singles = np.flatnonzero(n == 1)
    u = None
    if not len(singles):
        distinct_means = (np.diag(totals) - same_author) / (n * (n - 1))
        correction = distinct_means - np.diag(means)
        u = v + h.T @ (correction[:, None] * h) / 1024
    return {
        "counts": n,
        "V": v,
        "D": diagonal,
        "R": v - diagonal,
        "U": u,
        "singleton_profiles": singles,
        "profile_pair_means": means,
    }


def cyclic_pairing(codes):
    """Construct one explicit within-profile identity counterfactual, not new data."""
    c = np.asarray(codes)
    if c.ndim != 1 or not np.isin(c, np.arange(32)).all():
        raise ValueError("Valid full-profile addresses required")
    permutation = np.arange(len(c))
    for g in range(32):
        ids = np.flatnonzero(c == g)
        if len(ids) < 2:
            raise ValueError("Full re-pairing needs two authors in every profile")
        permutation[ids] = np.roll(ids, 1)
    return permutation


def literal_diagonal(kernel, codes):
    """Independently enumerate within-profile actual/distinct-index contributions."""
    k, c = np.asarray(kernel, float), np.asarray(codes)
    h = characters()
    n = np.bincount(c, minlength=32)
    if n.min() == 0:
        raise ValueError("All32 profiles required")
    v = np.zeros(32)
    d = np.zeros(32)
    u = np.zeros(32) if n.min() >= 2 else None
    for g in range(32):
        gi = np.flatnonzero(c == g)
        for b in range(32):
            bi = np.flatnonzero(c == b)
            ordinary = actual = distinct = 0.0
            for i in gi:
                for j in bi:
                    value = float(k[i, j])
                    ordinary += value
                    if i == j:
                        actual += value
                    else:
                        distinct += value
            signs = h[g] * h[b] / 1024
            v += signs * ordinary / (n[g] * n[b])
            d += signs * actual / (n[g] * n[b])
            if u is not None:
                moment = (
                    distinct / (n[g] * (n[g] - 1))
                    if g == b
                    else ordinary / (n[g] * n[b])
                )
                u += signs * moment
    return {"V": v, "D": d, "R": v - d, "U": u}
