#!/usr/bin/env python3
"""Finite author-observation and intervention-response accounts, not prediction."""

from __future__ import annotations

import numpy as np


def matrix(value):
    """Require a nonempty finite two-dimensional observation table."""
    x = np.asarray(value, float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Nonempty finite matrix required")
    return x


def choice_columns(groups):
    """Keep all registered condition/action masses rather than subtracting them."""
    g = np.asarray(groups)
    if g.ndim != 1 or not len(g):
        raise ValueError("Registered event-group addresses required")
    return np.equal(g[:, None], np.unique(g)[None, :]).astype(float)


def observation_space(observations):
    """Construct a fixed numerical column-space projector with no outcome tuning."""
    b = matrix(observations)
    scale = np.linalg.norm(b, axis=0)
    scaled = b / np.where(scale > 0, scale, 1)
    u, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    tolerance = 100 * max(scaled.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    q = u[:, :rank]
    return q, {
        "rank": rank,
        "tolerance": float(tolerance),
        "singular": singular.tolist(),
    }


def response_account(observations, response):
    """Separate response moments identified by observations from their lost part."""
    b, d = matrix(observations), matrix(response)
    if b.shape[0] != d.shape[0]:
        raise ValueError("Observation and response evidence supports differ")
    q, metadata = observation_space(b)
    visible = q @ (q.T @ d)
    hidden = d - visible
    return visible, hidden, metadata


def positive_witnesses(hidden, fraction=0.4):
    """Return every fixed-channel positive +/- witness, including exact nulls."""
    h = matrix(hidden)
    if not 0 < fraction < 1:
        raise ValueError("Strictly interior probability fraction required")
    size = np.max(abs(h), axis=0)
    active = size > 1e-12
    direction = np.zeros_like(h)
    direction[:, active] = h[:, active] / size[active]
    uniform = np.full_like(h, 1 / len(h))
    return np.stack(
        (
            uniform + fraction / len(h) * direction,
            uniform - fraction / len(h) * direction,
        )
    ), active


def source_observers(native, local, additive, groups):
    """Supply four fixed moment dictionaries, preserving condition and paired units."""
    n, g, a = map(matrix, (native, local, additive))
    if n.shape != a.shape or n.shape[0] != g.shape[0] or len(groups) != len(n):
        raise ValueError("All observers must use the same qualified evidence atoms")
    c = choice_columns(groups)
    conditional = (c[:, :, None] * n[:, None]).reshape(len(n), -1)
    return {
        "native": np.column_stack((c, n)),
        "native_local": np.column_stack((c, n, g)),
        "native_paired": np.column_stack((c, n, a)),
        "condition_native": np.column_stack((c, conditional)),
    }


def cold_scores(observations, groups, histories):
    """Score explicit event records without laws, truth, interventions or text.

    Atom addresses identify complete registered observations. Duplicate addresses
    are repeated sampled events here, not an implicit duplicate-report ledger.
    Histories must declare unique event IDs upstream before this numeric interface.
    """
    x, ids = matrix(observations), np.asarray(histories)
    groups = np.asarray(groups)
    if (
        ids.ndim != 2
        or min(ids.shape) == 0
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids >= len(x))
        or len(groups) != len(x)
    ):
        raise ValueError(
            "Complete integer event addresses on registered support required"
        )
    c = choice_columns(groups)
    counts = np.array([np.bincount(row, minlength=len(x)) for row in ids])
    masses = counts @ c
    totals = counts @ x
    conditional_totals = np.einsum("pe,eg,ed->pgd", counts, c, x)
    means = np.full_like(conditional_totals, np.nan, dtype=float)
    np.divide(
        conditional_totals, masses[:, :, None], out=means, where=masses[:, :, None] > 0
    )
    return {
        "mean": totals / ids.shape[1],
        "counts": masses,
        "conditional_mean": means,
        "event_counts": counts,
    }
