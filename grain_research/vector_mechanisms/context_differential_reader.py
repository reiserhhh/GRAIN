#!/usr/bin/env python3
"""Source-normalized conditional vector readers and their directional derivative."""

import numpy as np

from . import endpoint_query_contrast as contrast
from . import source_endpoint_assembly as endpoint

VARIANTS = ("constant", "context_kernel")


def build(bank, members):
    """Construct one source reference, indexing membership before any estimation.

    Each selected bank entry has a paired-context vector and two query readers.
    Nonmembers are never inspected. No recipient-dependent parameter is fitted.
    """
    members = np.asarray(members, int)
    if members.ndim != 1 or len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("At least two distinct source members required")
    rows = [bank[int(i)] for i in members]
    c, r = (
        np.asarray([x[0] for x in rows], float),
        np.asarray([x[1] for x in rows], float),
    )
    if c.ndim != 2 or r.shape != (len(c), 2, c.shape[1]):
        raise ValueError("Aligned source context and two readers required")
    if not np.isfinite(c).all() or not np.isfinite(r).all():
        raise ValueError("Finite source arrays required")
    ds = np.sum((c[:, None] - c[None, :]) ** 2, axis=-1)
    h2 = float(np.median(ds[np.triu_indices(len(c), 1)]))
    return {
        "c": c,
        "m": r.mean(axis=1).mean(axis=0),
        "d": r[:, 0] - r[:, 1],
        "h2": np.asarray(h2),
        "members": members,
    }


def weights(model, c, conditional=True):
    """Stable normalized Gaussian weights, with a declared uniform degeneracy."""
    c = np.asarray(c, float)
    if c.shape != model["c"].shape[1:] or not np.isfinite(c).all():
        raise ValueError("One finite aligned context required")
    h2 = float(model["h2"])
    distance2 = np.sum((model["c"] - c) ** 2, axis=1)
    if not conditional or h2 == 0:
        p = np.full(len(distance2), 1 / len(distance2))
    else:
        logits = -distance2 / (2 * h2)
        p = np.exp(logits - logits.max())
        p /= p.sum()
    return p, distance2


def read(model, c, conditional=True):
    """Reconstruct two query readers from a shared mean and conditional difference."""
    p, _ = weights(model, c, conditional)
    d = p @ model["d"]
    return np.stack((model["m"] + d / 2, model["m"] - d / 2))


def derivative(model, c, dc, conditional=True):
    """Full reader derivative for a context direction; do not stop-gradient c."""
    dc = np.asarray(dc, float)
    if dc.shape != model["c"].shape[1:] or not np.isfinite(dc).all():
        raise ValueError("One finite context direction required")
    p, _ = weights(model, c, conditional)
    if not conditional or float(model["h2"]) == 0:
        return np.zeros((2, model["d"].shape[1]))
    dp = p * ((model["c"] - p @ model["c"]) @ dc) / float(model["h2"])
    dd = dp @ model["d"]
    return np.stack((dd / 2, -dd / 2))


def project(primal, vectors, r):
    """Contract both query modes and preserve the exact symmetric contrast terms.

    Inputs [query,arm,hidden], [query,arm,component,factor,hidden], [query,hidden].
    """
    modes = endpoint.readers(r)
    return dict(
        scalar=np.einsum("qah,qmh->qam", primal, modes),
        actions=np.einsum("qacfh,qmh->qacmf", vectors, modes),
        **contrast.split_packets(primal, vectors, r),
    )


def range_account(model, native_d, predicted_d):
    """Evaluation-only affine-span bound; never return recipient coefficients.

    All normalized mixtures lie in mean(d) + span(d_i-mean(d)). The orthogonal
    target residual lower-bounds vector SSE, not its projection through a reader.
    """
    d = model["d"]
    center = d.mean(axis=0)
    _, s, vt = np.linalg.svd(d - center, full_matrices=False)
    tol = s[0] * max(d.shape) * np.finfo(float).eps if s[0] else 0
    basis = vt[s > tol]
    z = np.asarray(native_d) - center
    orth = z - (z @ basis.T) @ basis
    error = np.sum((np.asarray(predicted_d) - native_d) ** 2)
    lower = float(orth @ orth)
    if lower > error + 1e-9:
        raise ValueError("Mixture escaped its declared source affine span")
    return {
        "rank": len(basis),
        "native_norm2": float(np.sum(native_d**2)),
        "vector_error2": float(error),
        "orthogonal_lower_bound2": lower,
        "in_span_error2": float(max(0, error - lower)),
    }


def diagnostics(model, c):
    """Describe source support only; no tuning or recipient-target access."""
    p, ds = weights(model, c)
    h2 = float(model["h2"])
    return {
        "entropy": float(-np.sum(p[p > 0] * np.log(p[p > 0]))),
        "effective_sources": float(1 / (p @ p)),
        "largest_weight": float(p.max()),
        "smallest_weight": float(p.min()),
        "nearest_distance2_over_h2": float(ds.min() / h2) if h2 else 0.0,
        "mean_distance2_over_h2": float(ds.mean() / h2) if h2 else 0.0,
        "degenerate_support": int(h2 == 0),
    }
