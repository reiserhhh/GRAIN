#!/usr/bin/env python3
"""HM36 finite source-observed covector support in the fixed physical metric."""

from __future__ import annotations

import numpy as np


def source_basis(columns, roles, rank_multiplier=100.0):
    """Build a thin source-only basis, retaining scales and a fixed Gram rank rule."""
    x = np.asarray(columns, dtype=np.float64)
    if x.ndim != 2 or len(roles) != x.shape[1] or not np.isfinite(x).all():
        raise ValueError("finite columns with one explicit role each required")
    if any(role != "source" for role in roles):
        raise ValueError("application columns cannot enter source construction")
    scales = np.linalg.norm(x, axis=0)
    keep = scales > 0
    normalized = x[:, keep] / scales[keep]
    gram = normalized.T @ normalized
    eigenvalues, vectors = np.linalg.eigh(gram)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    tolerance = rank_multiplier * np.finfo(float).eps * max(gram.shape) * largest
    if eigenvalues.size and eigenvalues[0] < -tolerance:
        raise ValueError("Gram has a negative eigenvalue beyond fixed tolerance")
    retained = eigenvalues > tolerance
    rank = int(retained.sum())
    if rank:
        initial = normalized @ (vectors[:, retained] / np.sqrt(eigenvalues[retained]))
        basis, _ = np.linalg.qr(initial, mode="reduced")
    else:
        basis = np.empty((x.shape[0], 0))
    return {
        "basis": basis,
        "scales": scales,
        "nonzero": keep,
        "eigenvalues": eigenvalues,
        "rank": rank,
        "tolerance": tolerance,
        "rank_multiplier": rank_multiplier,
    }


def random_basis(dimension, rank, seed):
    """Draw one predeclared dimension-matched control, without data dependence."""
    if not 0 <= rank <= dimension:
        raise ValueError("rank outside ambient dimension")
    if rank == 0:
        return np.empty((dimension, 0))
    matrix = np.random.default_rng(seed).standard_normal((dimension, rank))
    return np.linalg.qr(matrix, mode="reduced")[0]


def action_parts(basis, covectors, displacement):
    """Split signed actions; geometric orthogonality does not remove cancellation."""
    u = np.asarray(basis, float)
    v = np.asarray(covectors, float)
    d = np.asarray(displacement, float).ravel()
    if v.ndim != 2 or v.shape[1] != d.size or u.shape[0] != d.size:
        raise ValueError("aligned thin basis, covectors and displacement required")
    dc, vc = u.T @ d, v @ u
    full = v @ d
    supported = vc @ dc
    missing = full - supported
    return {
        "full": full,
        "supported": supported,
        "missing": missing,
        "displacement_energy": float(d @ d),
        "displacement_supported_energy": float(dc @ dc),
        "covector_energy": np.sum(v * v, axis=1),
        "covector_supported_energy": np.sum(vc * vc, axis=1),
    }


def metric_projection(columns, vector, metric):
    """Small synthetic/reference metric projection, including nonorthogonal charts."""
    x, d, g = (np.asarray(z, float) for z in (columns, vector, metric))
    return x @ np.linalg.solve(x.T @ g @ x, x.T @ g @ d)


def signed_metrics(rows):
    """Report signed energy and relative missing action without clipping ratios."""
    a = np.asarray(rows, float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("full/supported/missing triples required")
    full, supported, missing = a.T
    ef, es, em = (float(np.mean(z * z)) for z in (full, supported, missing))
    cross = float(2 * np.mean(supported * missing))
    return {
        "count": len(a),
        "full_rms": float(np.sqrt(ef)),
        "supported_rms": float(np.sqrt(es)),
        "missing_rms": float(np.sqrt(em)),
        "supported_over_full": float(np.sqrt(es / ef)) if ef else None,
        "missing_over_full": float(np.sqrt(em / ef)) if ef else None,
        "signed_cross_mean": cross,
        "cross_over_full_energy": cross / ef if ef else None,
        "identity_error": abs(ef - es - em - cross),
    }
