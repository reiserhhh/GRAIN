#!/usr/bin/env python3
"""Fixed response-sector geometry, with no learned alignment or accuracy search."""

from __future__ import annotations

import itertools

import numpy as np


def characters(cube):
    """Expand a complete first-two-axis b/c cube into constant,b,c,bc sectors."""
    a = np.asarray(cube, dtype=float)
    if a.shape[:2] != (2, 2) or not np.isfinite(a).all():
        raise ValueError("Finite complete b/c cube required")
    signs = np.array([-1.0, 1.0])
    weights = np.stack(
        [
            np.ones((2, 2)),
            signs[:, None] * np.ones((1, 2)),
            np.ones((2, 1)) * signs[None, :],
            signs[:, None] * signs[None, :],
        ]
    )
    return np.tensordot(weights / 4, a, axes=((1, 2), (0, 1)))


def reconstruct(coefficients):
    """Recover all four original corners without selecting a sector."""
    a = np.asarray(coefficients, dtype=float)
    if a.shape[0] != 4 or not np.isfinite(a).all():
        raise ValueError("Four finite character coefficients required")
    return np.stack(
        [
            np.stack([a[0] + b * a[1] + c * a[2] + b * c * a[3] for c in (-1, 1)])
            for b in (-1, 1)
        ]
    )


def centered_gram(matrix):
    """Use declared Euclidean coordinates, centering rows but fitting nothing."""
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Finite row-by-coordinate matrix required")
    centered = a - a.mean(axis=0, keepdims=True)
    return centered @ centered.T


def gram_alignment(left, right):
    """CKA is undefined for a zero centered field; never manufacture agreement."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("Equal square Gram matrices required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite Gram matrices required")
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if scale == 0 else float(np.sum(a * b) / scale)


def finite_pairings(left, right):
    """Compare all family permutations; ranks describe this finite design only."""
    a, b = centered_gram(left), centered_gram(right)
    if a.shape != b.shape or len(a) > 8:
        raise ValueError("Matched support of at most eight families required")
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale == 0:
        return {
            "cka": None,
            "permutations": 0,
            "matching_rank_with_ties": None,
            "finite_pairing_tail_fraction": None,
            "pairing_mean": None,
            "matched_minus_pairing_mean": None,
        }, np.empty(0)
    values = np.array(
        [
            np.sum(a * b[np.ix_(p, p)]) / scale
            for p in itertools.permutations(range(len(a)))
        ]
    )
    actual = gram_alignment(a, b)
    tail = int(np.count_nonzero(values >= actual - 1e-12))
    return {
        "cka": actual,
        "permutations": len(values),
        "matching_rank_with_ties": tail,
        "finite_pairing_tail_fraction": tail / len(values),
        "pairing_mean": float(values.mean()),
        "matched_minus_pairing_mean": float(actual - values.mean()),
    }, values


def account_gap(cube, strength=0.4):
    """Direct difference of positive joint laws with identical b/c margins."""
    if not 0 < strength < 1:
        raise ValueError("Strictly positive probability laws required")
    a = np.asarray(cube, float)
    characters(a)
    joint = np.array([[1.0, -1.0], [-1.0, 1.0]])
    plus, minus = (1 + strength * joint) / 4, (1 - strength * joint) / 4
    return np.tensordot(plus - minus, a, axes=((0, 1), (0, 1)))
