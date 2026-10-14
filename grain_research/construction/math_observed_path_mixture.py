#!/usr/bin/env python3
"""Describe author-mixture path dependence without refitting observation laws."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_observed_path_memory as M


def mixture_covariance(mass, targets):
    """Express mixture minus pooled Markov mass as conditional author covariance."""
    p = np.asarray(mass, float)
    profiles, ix, full = M.aggregate(p, targets)
    middle = p.sum((1, 3))
    left = np.divide(
        p.sum(3),
        middle[:, None, :],
        out=np.zeros_like(p.sum(3)),
        where=middle[:, None, :] > 0,
    )
    right = np.divide(
        p.sum(1),
        middle[:, :, None],
        out=np.zeros_like(p.sum(1)),
        where=middle[:, :, None] > 0,
    )
    mixture = np.zeros_like(full)
    np.add.at(mixture, ix, M.markov_table(p) / len(p))
    difference = np.zeros_like(full)
    for g in range(len(profiles)):
        used = ix == g
        mid = middle[used]
        weights = np.divide(
            mid,
            mid.sum(0),
            out=np.zeros_like(mid),
            where=mid.sum(0) > 0,
        )
        mean_left = np.einsum("ub,uab->ab", weights, left[used])
        mean_right = np.einsum("ub,ubc->bc", weights, right[used])
        covariance = np.einsum("ub,uab,ubc->abc", weights, left[used], right[used])
        covariance -= mean_left[:, :, None] * mean_right[None, :, :]
        difference[g] = covariance * (mid.sum(0) / len(p))[None, :, None]
    actual = mixture - M.markov_table(full)
    return {
        "profiles": profiles,
        "covariance_mass": difference,
        "mixture_minus_pooled": actual,
        "identity_error": float(np.max(abs(difference - actual))),
    }


def kl_decomposition(observed, mixture, pooled):
    """Retain the signed cross term in a three-law KL account."""
    f, j, i = (np.asarray(v, float) for v in (observed, mixture, pooled))
    if (
        f.shape != j.shape
        or f.shape != i.shape
        or any(not np.isfinite(v).all() or np.any(v < 0) for v in (f, j, i))
    ):
        raise ValueError("Aligned nonnegative finite laws required")
    total, remaining, between = (
        M.divergence(f, i),
        M.divergence(f, j),
        M.divergence(j, i),
    )
    active = j > 0
    cross = float(np.sum((f[active] - j[active]) * np.log(j[active] / i[active])))
    return {
        "total_KL_nats": total,
        "remaining_KL_nats": remaining,
        "mixture_KL_nats": between,
        "signed_cross_nats": cross,
        "identity_error": abs(total - remaining - between - cross),
    }


def field_decomposition(left, right, weights):
    """Decompose paired field recurrence on one shared observed support."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.shape != right.shape or left.shape[0:2] != (3, 5):
        raise ValueError("Three laws and five aligned target fields required")
    lw, rw = left[2] - left[1], right[2] - right[1]
    lb, rb = left[1] - left[0], right[1] - right[0]
    common = np.isfinite(left).all((0, 1)) & np.isfinite(right).all((0, 1))
    reference = np.where(common, weights, 0.0)
    terms = np.array(
        [
            M.recurrence(lw + lb, rw + rb, reference)[0],
            M.recurrence(lw, rw, reference)[0],
            M.recurrence(lb, rb, reference)[0],
            M.recurrence(lw, rb, reference)[0],
            M.recurrence(lb, rw, reference)[0],
        ]
    )
    known = np.isfinite(terms).all(0)
    error = (
        float(np.max(abs(terms[0, known] - terms[1:, known].sum(0))))
        if known.any()
        else None
    )
    return {
        "terms": terms,
        "identity_error": error,
        "reference_mass_coverage": float(reference.sum()),
    }
