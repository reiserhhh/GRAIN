#!/usr/bin/env python3
"""Exact event/within-event changes of a finite positive observation measure."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _inputs(log_base, potential, groups):
    """Validate finite head-by-token logits and nonempty canonical groups."""
    base, b = np.asarray(log_base, float), np.asarray(potential, float)
    ids = np.asarray(groups)
    if base.ndim != 2 or base.shape != b.shape or not base.size:
        raise ValueError("Compatible nonempty head/token arrays required")
    if not np.isfinite(base).all() or not np.isfinite(b).all():
        raise ValueError("Finite potentials required")
    if ids.shape != (base.shape[1],) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Integer group for every token required")
    levels = np.unique(ids)
    if not np.array_equal(levels, np.arange(len(levels))):
        raise ValueError("Nonempty consecutive event groups required")
    return base, b, ids, levels


def normalize(logits):
    """Return strictly positive log probabilities without exponential overflow."""
    x = np.asarray(logits, float)
    return x - logsumexp(x, axis=-1, keepdims=True)


def shared_specific(potentials):
    """Split this fixed target battery into a common logit gauge and deviations."""
    b = np.asarray(potentials, float)
    if b.ndim != 3 or not b.size or not np.isfinite(b).all():
        raise ValueError("Finite target/head/token potentials required")
    centered = b - b.mean(-1, keepdims=True)
    shared = centered.mean(0)
    return shared, centered - shared


def factorize(log_base, potential, groups):
    """Factor an exact tilt into event probabilities and within-event token laws.

    Event-only keeps the original conditional token law; within-only keeps
    original event mass. Their joint reconstruction equals the full tilt.
    Mean-zero logits alone do not impose either constraint.
    """
    base, b, ids, levels = _inputs(log_base, potential, groups)
    p0, p1 = normalize(base), normalize(base + b)
    mass0 = np.stack([logsumexp(p0[:, ids == g], axis=1) for g in levels], axis=1)
    mass1 = np.stack([logsumexp(p1[:, ids == g], axis=1) for g in levels], axis=1)
    conditional0 = p0 - mass0[:, ids]
    conditional1 = p1 - mass1[:, ids]
    event_only = mass1[:, ids] + conditional0
    within_only = mass0[:, ids] + conditional1
    between_kl = np.sum(np.exp(mass1) * (mass1 - mass0), axis=1)
    conditional_kl = np.stack(
        [
            np.sum(
                np.exp(conditional1[:, ids == g])
                * (conditional1[:, ids == g] - conditional0[:, ids == g]),
                axis=1,
            )
            for g in levels
        ],
        axis=1,
    )
    within_kl = np.sum(np.exp(mass1) * conditional_kl, axis=1)
    total_kl = np.sum(np.exp(p1) * (p1 - p0), axis=1)
    u = np.stack(
        [logsumexp(conditional0[:, ids == g] + b[:, ids == g], axis=1) for g in levels],
        axis=1,
    )
    naive = b.copy()
    for g in levels:
        ix = ids == g
        naive[:, ix] -= np.sum(np.exp(conditional0[:, ix]) * b[:, ix], axis=1)[:, None]
    naive_log = normalize(base + naive)
    naive_mass = np.stack(
        [np.exp(naive_log[:, ids == g]).sum(1) for g in levels], axis=1
    )
    return {
        "cold": p0,
        "native": p1,
        "event_only": event_only,
        "within_only": within_only,
        "mass0": np.exp(mass0),
        "mass1": np.exp(mass1),
        "event_log_tilt": u,
        "within_log_tilt": b - u[:, ids],
        "between_kl": between_kl,
        "within_kl": within_kl,
        "total_kl": total_kl,
        "naive_mass_drift": np.max(np.abs(naive_mass - np.exp(mass0)), axis=1),
    }


def crossed_components(cold, first, second, full):
    """Symmetric two-component action account, retaining the nonlinear interaction."""
    c, a, b, f = (np.asarray(x, float) for x in (cold, first, second, full))
    return (a - c + f - b) / 2, (b - c + f - a) / 2, f - a - b + c
