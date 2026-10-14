#!/usr/bin/env python3
"""Retain context mass and distinguish common action from program discrepancy."""

from __future__ import annotations

import numpy as np


def context_account_features(coordinates, context):
    """Return unnormalized joint context mass/means, never subtracting context."""
    g, labels = np.asarray(coordinates, float), np.asarray(context)
    if g.ndim != 2 or labels.shape != (len(g),) or not np.isfinite(g).all():
        raise ValueError(
            "Finite event matrix and one discrete context per event required"
        )
    if labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Nonnegative integer context IDs required")
    levels = np.unique(labels)
    indicators = (labels[:, None] == levels[None, :]).astype(float)
    payload = np.column_stack([np.ones(len(g)), g])
    return (indicators[:, :, None] * payload[:, None, :]).reshape(len(g), -1)


def constraint_fiber(observations, contrasts, rtol=1e-10):
    """Compute one fixed constraint kernel with row-scale invariance and diagnostics."""
    x, d = np.asarray(observations, float), np.asarray(contrasts, float)
    if x.ndim != 2 or d.ndim != 2 or len(x) != len(d):
        raise ValueError("Matched event rows required")
    if not np.isfinite(x).all() or not np.isfinite(d).all() or rtol <= 0:
        raise ValueError("Finite values and positive fixed cutoff required")
    raw = (d.T @ x).T
    norms = np.linalg.norm(raw, axis=1)
    matrix = raw[norms > 0] / norms[norms > 0, None]
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    cutoff = rtol * singular[0] if len(singular) else 0.0
    rank = int(np.count_nonzero(singular > cutoff))
    raw_singular = np.linalg.svd(raw, compute_uv=False)
    raw_cutoff = rtol * raw_singular[0] if len(raw_singular) else 0.0
    raw_rank = int(np.count_nonzero(raw_singular > raw_cutoff))
    basis = vh[rank:].T
    return {
        "rank": rank,
        "nullity": d.shape[1] - rank,
        "raw_rank": raw_rank,
        "singular_values": singular,
        "raw_singular_values": raw_singular,
        "cutoff": float(cutoff),
        "basis": basis,
        "normalized_constraints": matrix,
    }


def observer_contract(native, name):
    """Specify agreement/fact constraints without inspecting their success or size."""
    y = np.asarray(native, float)
    if y.ndim != 5 or y.shape[1:] != (2, 2, 7, 2) or not np.isfinite(y).all():
        raise ValueError("event/program/actor/query/unit array required")
    definitions = {
        "A_traits_score": (range(1), range(6), range(1), False),
        "A_traits_score_fact_score": (range(1), range(6), range(1), True),
        "A_traits_both_units": (range(1), range(6), range(2), False),
        "A_traits_both_units_fact_both_units": (range(1), range(6), range(2), True),
        "AB_all_both_units": (range(2), range(7), range(2), False),
    }
    if name not in definitions:
        raise ValueError("Unknown fixed observer contract")
    actors, queries, units, facts = definitions[name]
    agreed = [(a, q, u) for a in actors for q in queries for u in units]
    columns = [y[:, 0, a, q, u] - y[:, 1, a, q, u] for a, q, u in agreed]
    neutral = [(r, 0, 6, u) for r in range(2) for u in units] if facts else []
    columns += [y[:, r, a, q, u] for r, a, q, u in neutral]
    return np.column_stack(columns), agreed, neutral
