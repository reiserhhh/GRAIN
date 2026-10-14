#!/usr/bin/env python3
"""Finite conditional source-anchor assignment law, without response fitting."""

import numpy as np

from grain_research.construction import math_profile_reader as P

VERSION = "HM212_source_anchor_exchange_v1"


def design(labels, target):
    """Keep all other-four rows and all32 profile counts fixed."""
    y = np.asarray(labels)
    b = P.profile_codes(y, target).astype(int)
    counts = np.array([np.bincount(b[y[:, target] == g], minlength=16) for g in (0, 1)])
    if counts.min() == 0:
        raise ValueError("Unobserved anchor side; no zero imputation or smoothing")
    omega = counts.prod(0) / counts.sum(0)
    omega /= omega.sum()
    coefficients = np.where(
        y[:, target] == 1, omega[b] / counts[1, b], -omega[b] / counts[0, b]
    )
    return b, counts, omega, coefficients


def finite_projection(values, labels):
    """Require author-linked finite columns; views belong to one author row."""
    t = np.asarray(values, float)
    if t.ndim != 2 or len(t) != len(labels) or not np.isfinite(t).all():
        raise ValueError("Finite projected author values required")
    return t


def statistic(values, labels, target):
    """Read the unchanged matched contrast in an arbitrary projected space."""
    t = finite_projection(values, labels)
    return design(labels, target)[3] @ t


def exact_moments(values, labels, target, joint=False):
    """Return exact orbit mean and variance/covariance, not a normal test."""
    t = finite_projection(values, labels)
    b, counts, omega, _ = design(labels, target)
    result = np.zeros((t.shape[1], t.shape[1]) if joint else t.shape[1])
    for c in range(16):
        rows = t[b == c]
        centered = rows - rows.mean(0)
        scale = omega[c] ** 2 * len(rows) / (counts[:, c].prod() * (len(rows) - 1))
        result += scale * (centered.T @ centered if joint else (centered**2).sum(0))
    return np.zeros(t.shape[1]), result


def assignment_indices(labels, target, draws, seed):
    """Draw one within-stratum author permutation shared across all views."""
    if type(draws) is not int or draws < 1:
        raise ValueError("A positive fixed number of references required")
    b, _, _, _ = design(labels, target)
    rng = np.random.default_rng(np.random.SeedSequence([seed, target]))
    result = np.tile(np.arange(len(labels)), (draws, 1))
    for row in result:
        for c in range(16):
            ix = np.flatnonzero(b == c)
            row[ix] = rng.permutation(ix)
    return result


def validate_assignments(labels, target, indices):
    """Reject cross-stratum transport, duplicate rows or noninteger indices."""
    ix = np.asarray(indices)
    b, _, _, _ = design(labels, target)
    if (
        ix.ndim != 2
        or ix.shape[1] != len(labels)
        or not np.issubdtype(ix.dtype, np.integer)
        or not np.array_equal(
            np.sort(ix, axis=1), np.tile(np.arange(len(b)), (len(ix), 1))
        )
        or not np.array_equal(b[ix], np.tile(b, (len(ix), 1)))
    ):
        raise ValueError("Complete within-stratum author permutations required")
    return ix


def references(values, labels, target, indices):
    """Apply frozen paired-view assignments without refitting any native response."""
    t = finite_projection(values, labels)
    ix = validate_assignments(labels, target, indices)
    return design(labels, target)[3][ix] @ t


def rank_account(observed, sampled, variance):
    """Keep signed effects and undefined degenerate ranks separate."""
    s = np.asarray(sampled, float)
    if (
        s.ndim != 1
        or len(s) < 1
        or not np.isfinite(s).all()
        or not np.isfinite([observed, variance]).all()
        or variance < 0
    ):
        raise ValueError("Finite scalar observation and orbit variance required")
    if variance == 0:
        return {
            "orbit_sd": 0.0,
            "standardized": None,
            "upper_rank": None,
            "two_sided_rank": None,
        }
    return {
        "orbit_sd": float(np.sqrt(variance)),
        "standardized": float(observed / np.sqrt(variance)),
        "upper_rank": float((1 + np.count_nonzero(s >= observed)) / (len(s) + 1)),
        "two_sided_rank": float(
            (1 + np.count_nonzero(abs(s) >= abs(observed))) / (len(s) + 1)
        ),
    }


def holm(values):
    """Retain the declared family size when any member has undefined evidence."""
    p = np.array([1.0 if x is None else x for x in values])
    if not np.isfinite(p).all() or np.min(p) < 0 or np.max(p) > 1:
        raise ValueError("Ranks must lie in[0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    )
    return [None if x is None else float(adjusted[i]) for i, x in enumerate(values)]
