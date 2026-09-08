#!/usr/bin/env python3
"""Explicit input-embedding pullback candidates, without Transformer execution.

Each chart anchors two output derivatives at a construction text. A fixed
alignment kernel transports token embeddings to that chart. This approximation
tests whether static token geometry suffices; it does not presume that it does.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

METHODS = {"uniform": (0.0, 0.0), "semantic": (8.0, 0.0), "semantic_order": (8.0, 4.0)}


def pool_pairings(reference_embeddings, input_embeddings, gradients, method):
    """Contract aligned static embeddings with two frozen input Jacobian rows."""
    r, x, g = (
        np.asarray(v, dtype=np.float32)
        for v in (reference_embeddings, input_embeddings, gradients)
    )
    if (
        r.ndim != 2
        or x.ndim != 2
        or not len(r)
        or not len(x)
        or r.shape[1] != x.shape[1]
        or g.shape != (2, *r.shape)
    ):
        raise ValueError(
            "Two matched projection gradients and nonempty embeddings required"
        )
    if not all(np.isfinite(v).all() for v in (r, x, g)) or method not in METHODS:
        raise ValueError("Invalid independent operator input")
    tau, order = METHODS[method]
    if tau == 0 and order == 0:
        return g.sum(1) @ x.mean(0)
    ru = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
    xu = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    score = tau * (ru @ xu.T)
    if order:
        rp = np.linspace(0, 1, len(r), dtype=np.float32)
        xp = np.linspace(0, 1, len(x), dtype=np.float32)
        score -= order * (rp[:, None] - xp[None, :]) ** 2
    aligned = softmax(score, axis=1) @ x
    return np.einsum("ard,rd->a", g, aligned)


def projection_from_embeddings(chart, input_embeddings, method):
    """Rebase the alignment itself, guaranteeing exact identity at the chart origin."""
    r, g = chart["embeddings"], chart["gradients"]
    current = pool_pairings(r, input_embeddings, g, method)
    origin = chart.get("offsets", {}).get(method)
    if origin is None:
        origin = pool_pairings(r, r, g, method)
    return np.asarray(chart["projection_origin"], float) + current - origin


def score_token_ids(ids, mask, table, charts, anchor, method):
    """Score a new input token sequence using only a fixed embedding table and charts."""
    ids, mask = np.asarray(ids), np.asarray(mask, bool)
    if (
        ids.ndim != 1
        or not np.issubdtype(ids.dtype, np.integer)
        or ids.shape != mask.shape
        or not len(ids)
        or ids.min() < 0
        or ids.max() >= len(table)
        or not mask.any()
        or mask[-1]
    ):
        raise ValueError("Valid token IDs and query mask required")
    evidence = ~mask
    evidence[-1] = False
    x = np.asarray(table[ids[evidence]], dtype=np.float32)
    if not charts:
        raise ValueError("At least one frozen chart required")
    values = np.asarray([projection_from_embeddings(c, x, method) for c in charts])
    z = values.mean(0)
    return {
        "projection1": float(z[0]),
        "projection2": float(z[1]),
        "score": float(anchor + z.sum()),
        "charts": values.tolist(),
    }
