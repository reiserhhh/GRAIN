#!/usr/bin/env python3
"""Exact conditional-answer differential and source-referenced seed geometry."""

from __future__ import annotations

import numpy as np


def score_seed(logits, values, rows):
    """Differentiate a softmax expectation, then pull it back through LM-head rows."""
    z, v, w = (
        np.asarray(logits, float),
        np.asarray(values, float),
        np.asarray(rows, float),
    )
    if (
        z.ndim != 1
        or len(z) < 2
        or v.shape != z.shape
        or w.ndim != 2
        or len(w) != len(z)
        or not all(np.isfinite(x).all() for x in (z, v, w))
    ):
        raise ValueError(
            "Aligned finite candidate logits, values and output rows required"
        )
    p = np.exp(z - z.max())
    p /= p.sum()
    score = float(p @ v)
    b = p * (v - score)
    return {"p": p, "score": score, "candidate_seed": b, "hidden_seed": b @ w}


def row_span(rows, tolerance=1e-10):
    """Uncentered numerical span of source vectors; not a newly fitted predictor."""
    rows = np.asarray(rows, float)
    if rows.ndim != 2 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError("Finite nonempty source rows required")
    _, s, v = np.linalg.svd(rows, full_matrices=False)
    return v[s > tolerance * s[0]] if s[0] else v[:0]


def projection_fraction(vector, basis):
    """Report energy in a fixed source span, with zero-vector values undefined."""
    x, v = np.asarray(vector, float), np.asarray(basis, float)
    if x.ndim != 1 or v.ndim != 2 or v.shape[1] != len(x):
        raise ValueError("Vector and source span dimension mismatch")
    energy = float(x @ x)
    return float(np.sum((v @ x) ** 2) / energy) if energy else None


def binary_gain(probability):
    """The exact binary probability/logit-contrast scale in real arithmetic."""
    p = np.asarray(probability, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite probabilities in [0,1] required")
    return p * (1 - p)


def source_gains(scores, folds):
    """Use only each component's named source authors for a fixed mean gain."""
    gains = {}
    for held, sources in folds.items():
        if held in sources or len(sources) != 3 or len(set(sources)) != 3:
            raise ValueError("Three distinct non-held source authors required")
        gain = float(np.mean([binary_gain(scores[s]) for s in sources]))
        if gain <= 1e-12:
            raise ValueError("Source-only mean gain is numerically singular")
        gains[held] = gain
    return gains


def coefficient_actions(coefficients, basis, directions):
    """Contract density coefficients without materializing a full predicted field."""
    a, v, d = (np.asarray(x, float) for x in (coefficients, basis, directions))
    if (
        a.ndim != 2
        or v.ndim != 2
        or d.ndim != 3
        or a.shape != (d.shape[1], len(v))
        or d.shape[2] != v.shape[1]
    ):
        raise ValueError("Coefficient, direction and basis layout mismatch")
    return np.asarray([np.sum(a * (dk @ v.T)) / len(a) for dk in d])
