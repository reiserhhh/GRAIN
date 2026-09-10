#!/usr/bin/env python3
"""Numerical final-query attention channels with rotary positions and GQA."""

from __future__ import annotations

import numpy as np

from .attention_pullback import attention_pullback


def rotate_half(x):
    """Apply the paired half-turn used by the pinned Qwen2 rotary map."""
    x = np.asarray(x, float)
    if x.shape[-1] % 2:
        raise ValueError("Even rotary dimension required")
    a, b = np.split(x, 2, axis=-1)
    return np.concatenate([-b, a], axis=-1)


def rotary(x, cos, sin, *, transpose=False):
    """Apply a diagonal-cosine plus half-turn map, or its exact transpose."""
    x, cos, sin = (np.asarray(v, float) for v in (x, cos, sin))
    return (
        x * cos - rotate_half(x * sin) if transpose else x * cos + rotate_half(x) * sin
    )


def decompose_attention(q, k, v, g, cos, sin, wk, wv):
    """Return final-query value/key derivatives in normalized-input coordinates.

    q/g are [query_heads, head_dim]; k/v are [tokens, kv_heads, head_dim].
    All positions are admissible to the last query. Projection biases are
    already included in q/k/v; their input derivative uses only wk/wv.
    The direct query and residual routes at the final token are NOT included.
    """
    q, k, v, g, cos, sin, wk, wv = (
        np.asarray(x, float) for x in (q, k, v, g, cos, sin, wk, wv)
    )
    if (
        q.ndim != 2
        or k.ndim != 3
        or k.shape != v.shape
        or q.shape != g.shape
        or k.shape[2] != q.shape[1]
        or min(k.shape) < 1
        or q.shape[0] < 1
        or q.shape[0] % k.shape[1]
        or cos.shape != sin.shape
        or cos.shape != (k.shape[0], k.shape[2])
        or wk.ndim != 2
        or wk.shape != wv.shape
        or wk.shape[0] != k.shape[1] * k.shape[2]
        or not all(np.isfinite(x).all() for x in (q, k, v, g, cos, sin, wk, wv))
    ):
        raise ValueError("Aligned finite rotary/GQA arrays required")
    heads, d = q.shape
    groups = heads // k.shape[1]
    qr = rotary(q, cos[-1], sin[-1])
    kr = rotary(k, cos[:, None], sin[:, None])
    dk, dv = np.zeros_like(k), np.zeros_like(v)
    masses, contrasts, outputs = [], [], []
    for h in range(heads):
        kv = h // groups
        r = attention_pullback(qr[h], kr[:, kv], v[:, kv], g[h])
        dk[:, kv] += rotary(r["keys"], cos, sin, transpose=True)
        dv[:, kv] += r["values"]
        masses.append(r["attention"])
        contrasts.append(r["relative_value"])
        outputs.append(r["output"])
    return {
        "key_field": dk.reshape(len(k), -1) @ wk,
        "value_field": dv.reshape(len(v), -1) @ wv,
        "attention": np.stack(masses),
        "relative_value": np.stack(contrasts),
        "output": np.stack(outputs).reshape(heads * d),
    }
