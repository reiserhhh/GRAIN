#!/usr/bin/env python3
"""HM36 companion: Gram selects rank; stable QR/SVD constructs that source span."""

from __future__ import annotations

import numpy as np


def source_basis(columns, roles, rank_multiplier=100.0):
    """Retain the frozen rank criterion without inverse-square-root Gram vectors."""
    x = np.asarray(columns, dtype=np.float64)
    if x.ndim != 2 or len(roles) != x.shape[1] or not np.isfinite(x).all():
        raise ValueError("finite columns with one explicit role each required")
    if any(role != "source" for role in roles):
        raise ValueError("application columns cannot enter source construction")
    scales = np.linalg.norm(x, axis=0)
    keep = scales > 0
    normalized = x[:, keep] / scales[keep]
    gram = normalized.T @ normalized
    eigenvalues = np.linalg.eigvalsh(gram)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    tolerance = rank_multiplier * np.finfo(float).eps * max(gram.shape) * largest
    if eigenvalues.size and eigenvalues[0] < -tolerance:
        raise ValueError("Gram has a negative eigenvalue beyond fixed tolerance")
    rank = int((eigenvalues > tolerance).sum())
    if rank:
        q, r = np.linalg.qr(normalized, mode="reduced")
        left, singular, _ = np.linalg.svd(r, full_matrices=False)
        if int((singular**2 > tolerance).sum()) != rank:
            raise ValueError("stable singular values disagree with frozen Gram rank")
        basis = q @ left[:, :rank]
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


def alternate_basis(raw, metadata):
    """Direct tall-matrix SVD independently checks the companion's QR/SVD span."""
    scales = np.linalg.norm(raw, axis=0)
    matrix = raw[:, scales > 0] / scales[scales > 0]
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    selected = singular**2 > metadata["rank_tolerance"]
    if int(selected.sum()) != metadata["rank"]:
        raise ValueError("direct SVD disagrees with the frozen source rank")
    return left[:, selected]
