#!/usr/bin/env python3
"""Finite response-relation certificates independent of predictor accuracy."""

from __future__ import annotations

import numpy as np


def checked_matrix(values):
    """Require a finite nonempty observation matrix in declared numeric units."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Finite nonempty two-dimensional observations required")
    return x


def relation_projector(values):
    """Return the coefficient-space annihilator and fixed arithmetic rank."""
    x = checked_matrix(values)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    tol = np.finfo(float).eps * max(x.shape) * (s[0] if len(s) else 0)
    rank = int(np.count_nonzero(s > tol))
    q = (
        np.zeros((len(x), len(x)))
        if rank == len(x)
        else np.eye(len(x)) - u[:, :rank] @ u[:, :rank].T
    )
    return q, rank, s, float(tol)


def source_witness(projector, kernel_dimension):
    """Choose a source-only unit relation, or None when no relation exists."""
    q = checked_matrix(projector)
    if q.shape[0] != q.shape[1] or kernel_dimension < 0:
        raise ValueError("Square relation projector and nonnegative dimension required")
    if kernel_dimension == 0:
        return None
    j = int(np.argmax(np.diag(q)))
    c = q[:, j]
    norm = np.linalg.norm(c)
    if norm == 0:
        raise ArithmeticError("Nonempty kernel has no coordinate witness")
    return c / norm


def relation_certificate(source, target):
    """Measure finite violations of ker(source.T) subset ker(target.T)."""
    x, y = checked_matrix(source), checked_matrix(target)
    if len(x) != len(y):
        raise ValueError("Aligned observation rows required")
    q, rank, singular, tol = relation_projector(x)
    c = source_witness(q, len(x) - rank)
    defect = q @ y
    norm_y = np.linalg.norm(y)
    _, target_rank, _, _ = relation_projector(y)
    summary = {
        "rows": len(x),
        "source_dimensions": x.shape[1],
        "target_dimensions": y.shape[1],
        "source_rank": rank,
        "target_rank": target_rank,
        "source_kernel_dimension": len(x) - rank,
        "arithmetic_rank_threshold": tol,
        "arbitrary_target_transport_possible": rank == len(x),
        "arbitrary_invertible_extension_possible": rank == target_rank == len(x)
        and x.shape[1] == y.shape[1],
        "relative_relation_defect": float(np.linalg.norm(defect) / norm_y)
        if norm_y
        else 0.0,
        "source_projector_closure": float(np.linalg.norm(q @ x)),
        "has_source_witness": c is not None,
        "witness_source_rms": float(np.sqrt(np.mean((c @ x) ** 2)))
        if c is not None
        else None,
        "witness_target_rms": float(np.sqrt(np.mean((c @ y) ** 2)))
        if c is not None
        else None,
    }
    return summary, q, c, singular


def replicate_relation(c, neutral, styled):
    """Apply a fixed source relation to two unused generation replicates."""
    x, y = np.asarray(neutral, float), np.asarray(styled, float)
    c = np.asarray(c, float)
    if x.ndim != 3 or x.shape != y.shape or len(x) != 2 or x.shape[1] != len(c):
        raise ValueError("Two aligned held replicates and source coefficients required")
    if not all(np.isfinite(v).all() for v in (c, x, y)):
        raise ValueError("Finite relation inputs required")
    a, b = np.einsum("n,rnd->rd", c, x), np.einsum("n,rnd->rd", c, y)
    return {
        "neutral_rms": float(np.sqrt(np.mean(a**2))),
        "styled_rms": float(np.sqrt(np.mean(b**2))),
        "neutral_signed_cross_product": float(a[0] @ a[1]),
        "styled_signed_cross_product": float(b[0] @ b[1]),
        "change_signed_cross_product": float((b[0] - a[0]) @ (b[1] - a[1])),
    }
