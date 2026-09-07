#!/usr/bin/env python3
"""Exact head reconstruction and local projection from a frozen Qwen prefix."""

import numpy as np
import torch

from grain_research.vector_mechanisms.prefix import rotate_half
from grain_research.vector_mechanisms.query_tilt import tilted_readout


def full_head(cache):
    """Reconstruct all concatenated final-query heads with grouped-query KV."""
    q, k, v = (np.asarray(cache[name]) for name in ("q", "k", "v"))
    if (
        q.ndim != 2
        or k.ndim != 3
        or v.shape != k.shape
        or q.shape[1] != k.shape[2]
        or min(k.shape) == 0
        or len(q) == 0
        or len(q) % k.shape[1]
        or not all(np.isfinite(a).all() for a in (q, k, v))
    ):
        raise ValueError("Finite compatible query and grouped KV arrays required")
    ratio = len(q) // k.shape[1]
    return np.concatenate(
        [
            tilted_readout(k[:, h // ratio], v[:, h // ratio], query)[0]
            for h, query in enumerate(q)
        ]
    )


def projected_cache(observer, cache, groups, layer=17):
    """Reproject the exact S9 boundary locally; do not rerun any decoder block."""
    from scripts.run_grain_mean_null_path import mean_null_direction

    model = observer.model
    block = model.model.layers[layer]
    parameter = next(model.parameters())
    boundary = torch.as_tensor(
        cache["boundary"], device=parameter.device, dtype=parameter.dtype
    )
    groups = [list(g) for g in groups]
    if boundary.ndim != 2 or any(len(boundary) - 1 in g for g in groups):
        raise ValueError("Body groups must not include the final query token")
    with torch.no_grad():
        h0 = boundary + mean_null_direction(boundary, groups, list(range(len(groups))))
        if not torch.equal(h0[-1], boundary[-1]):
            raise ValueError("S9 projection changed the final query boundary")
        normalized = block.input_layernorm(h0[None])
        k = block.self_attn.k_proj(normalized)[0].detach().float().cpu().numpy()
        v = block.self_attn.v_proj(normalized)[0].detach().float().cpu().numpy()
        position_ids = torch.arange(len(boundary), device=parameter.device)[None]
        cos, sin = model.model.rotary_emb(h0[None], position_ids)
        cos, sin = (x[0].detach().float().cpu().numpy() for x in (cos, sin))
    if not np.array_equal(cos[-1], cache["last_cos"]) or not np.array_equal(
        sin[-1], cache["last_sin"]
    ):
        raise ValueError("Rotary position convention differs from native prefix")
    dim = block.self_attn.head_dim
    k, v = k.reshape(len(boundary), -1, dim), v.reshape(len(boundary), -1, dim)
    return {
        "q": cache["q"].copy(),
        "k": k * cos[:, None] + rotate_half(k) * sin[:, None],
        "v": v.copy(),
    }, {"query_boundary_error": 0.0, "query_vector_error": 0.0}


def configuration_scores(native, projected, reference):
    """Keep all views and apply the unchanged functional covector, without fitting."""
    native, projected, reference = (
        np.asarray(a, float) for a in (native, projected, reference)
    )
    if (
        native.ndim != 2
        or native.shape != projected.shape
        or reference.shape != (native.shape[1],)
        or not all(np.isfinite(a).all() for a in (native, projected, reference))
    ):
        raise ValueError("Finite paired view heads and one frozen covector required")
    delta = native - projected
    scores = delta @ reference
    return delta, scores, float(scores.mean())


def replay_heads(native, projected, old_native, old_projected, reference, old_score):
    """Check the shortcut against actual S24 heads and its frozen scalar score."""
    native_error = float(np.max(np.abs(native - old_native)))
    projected_error = float(np.max(np.abs(projected - old_projected)))
    score_error = abs(float(reference @ (native - projected)) - old_score)
    if (
        not np.isfinite([native_error, projected_error, score_error]).all()
        or max(native_error, projected_error) > 1e-4
        or score_error > 2e-5
    ):
        raise ValueError("Actual S24 head or frozen-score replay failed")
    return {
        "native_head_error": native_error,
        "projected_head_error": projected_error,
        "linear_score_error": score_error,
    }
