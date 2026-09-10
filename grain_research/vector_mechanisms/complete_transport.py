#!/usr/bin/env python3
"""Complete context transport from a cached decoder state to a final readout."""

from __future__ import annotations

import numpy as np
import torch

from .native_material_reader import direction
from .source_cotangent import source_means


def post_attention(model, state, kwargs, start=17):
    """Return final post-attention state and both residual/attention summands.

    Propagate all token positions through the actual remaining decoder blocks.
    Stop before the last MLP, final norm and language-model output projection.
    """
    if model.config.model_type != "qwen2" or not 0 <= start < len(model.model.layers):
        raise ValueError("Pinned Qwen2 suffix and valid start layer required")
    x = state
    for layer in model.model.layers[start:-1]:
        x = layer(x, **kwargs)
        x = x[0] if isinstance(x, tuple) else x
    last = model.model.layers[-1]
    attention, _ = last.self_attn(last.input_layernorm(x), **kwargs)
    direct, mediated = x[:, -1], attention[:, -1]
    return direct + mediated, direct, mediated


def tail_contrast(model, residual, token_ids, contrast):
    """Evaluate the declared fixed logit contrast after the last residual state."""
    layer = model.model.layers[-1]
    h = residual + layer.mlp(layer.post_attention_layernorm(residual))
    logits = model.lm_head(model.model.norm(h))
    return (logits[:, token_ids] * contrast).sum()


def capture_source_cotangent(model, state, kwargs, token_ids, contrast, start=17):
    """Capture t in the unchanged full-sequence source forward computation.

    Full-sequence MLP and the original final-token norm/head tensor shapes are
    retained to avoid a last-token GEMM rounding change in the pinned observer.
    """
    captured = {}

    def save_residual(_module, args):
        captured["residual"] = args[0]

    hook = model.model.layers[-1].post_attention_layernorm.register_forward_pre_hook(
        save_residual
    )
    try:
        with torch.enable_grad():
            h = state.detach().requires_grad_(True)
            for layer in model.model.layers[start:]:
                h = layer(h, **kwargs)
                h = h[0] if isinstance(h, tuple) else h
            logits = model.lm_head(model.model.norm(h[:, -1:]))[0, -1]
            score = (logits[token_ids] * contrast).sum()
            t = torch.autograd.grad(score, captured["residual"])[0][:, -1]
        return score.detach(), t.detach()
    finally:
        hook.remove()


def body_direction(state, grain, positions, factor):
    """Construct and freeze a GRAIN direction at the initial BODY positions only."""
    pos = np.asarray(positions)
    if (
        state.ndim != 3
        or state.shape[0] != 1
        or pos.ndim != 1
        or len(pos) < 2
        or not np.issubdtype(pos.dtype, np.integer)
        or np.any(np.diff(pos) <= 0)
        or pos.min() < 0
        or pos.max() >= state.shape[1] - 1
    ):
        raise ValueError("Ordered nonfinal BODY positions required")
    h = state[0, pos].detach().cpu().numpy().astype(float)
    delta, _ = direction(h, grain, factor)
    full = torch.zeros_like(state)
    full[:, pos] = torch.as_tensor(delta, dtype=state.dtype, device=state.device)
    return full


def fixed_means(sources, query, held=None):
    """Reuse the exact source membership rules for final-state cotangents."""
    means = source_means(
        {k: np.asarray(v)[None] for k, v in sources.items()}, query, held
    )
    return {"own_t": means["own_g"][0], "pooled_t": means["pooled_g"][0]}


def forward_response(model, state, kwargs, delta, start=17):
    """Compute a genuine forward-mode JVP, not a current-loss backward pass."""
    with torch.no_grad():
        primal, tangent = torch.func.jvp(
            lambda x: post_attention(model, x, kwargs, start), (state,), (delta,)
        )
    return tuple(x.detach() for x in primal), tuple(x.detach() for x in tangent)
