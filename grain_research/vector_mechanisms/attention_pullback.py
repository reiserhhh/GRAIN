#!/usr/bin/env python3
"""Exact one-query scaled-dot-product attention pullback, without model fitting."""

from __future__ import annotations

import numpy as np


def attention_pullback(query, keys, values, cotangent):
    """Split a scalar response into value transport and key/query selection.

    Inputs are one already-projected query, all admissible keys/values and a
    downstream output cotangent. Rotary maps, output projections, residuals,
    normalization and later layers require their own chain-rule operations.
    """
    q, k, v, g = (np.asarray(x, float) for x in (query, keys, values, cotangent))
    if (
        q.ndim != 1
        or k.ndim != 2
        or v.ndim != 2
        or g.ndim != 1
        or len(k) != len(v)
        or len(k) < 1
        or k.shape[1] != len(q)
        or v.shape[1] != len(g)
        or len(q) == 0
        or not all(np.isfinite(x).all() for x in (q, k, v, g))
    ):
        raise ValueError("Aligned finite query, keys, values and cotangent required")
    scale = np.sqrt(len(q))
    logits = k @ q / scale
    a = np.exp(logits - logits.max())
    a /= a.sum()
    output = a @ v
    contrast = (v - output) @ g
    score_gradient = a * contrast
    return {
        "attention": a,
        "output": output,
        "relative_value": contrast,
        "values": a[:, None] * g,
        "keys": score_gradient[:, None] * q[None] / scale,
        "query": score_gradient @ k / scale,
        "score_gradient": score_gradient,
    }
