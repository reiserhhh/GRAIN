#!/usr/bin/env python3
"""Signed residual-update transport and genuinely truncated GRAIN readers."""

from __future__ import annotations

import torch


def validate(model, start, exits):
    """Require ordered exits before the last block's MLP, without silent clipping."""
    last = len(model.model.layers) - 1
    if model.config.model_type != "qwen2" or not 0 <= start < last:
        raise ValueError("Pinned Qwen2 suffix required")
    if (
        not exits
        or tuple(sorted(set(exits))) != tuple(exits)
        or min(exits) < start
        or max(exits) >= last
    ):
        raise ValueError("Unique ordered earlier exits required")


def stage_labels(start=17, last=27):
    """Stable addresses for signed updates, not named psychological factors."""
    return [
        f"block{i:02d}_{kind}"
        for i in range(start, last)
        for kind in ("attention", "mlp")
    ] + [f"block{last:02d}_attention"]


def accumulate(model, state, kwargs, start=17, exits=(20, 23, 26)):
    """Return early/full endpoints and all additive last-position update vectors.

    Every update is evaluated with the actual upstream context, keeping all token
    positions. Differentiating the sum is telescoping, not separate ablation.
    """
    validate(model, start, exits)
    x, updates, selected = state, [], []
    for i in range(start, len(model.model.layers) - 1):
        layer = model.model.layers[i]
        attention, _ = layer.self_attn(layer.input_layernorm(x), **kwargs)
        x = x + attention
        mlp = layer.mlp(layer.post_attention_layernorm(x))
        x = x + mlp
        updates.extend((attention[:, -1], mlp[:, -1]))
        if i in exits:
            selected.append(x[:, -1])
    last = model.model.layers[-1]
    attention, _ = last.self_attn(last.input_layernorm(x), **kwargs)
    updates.append(attention[:, -1])
    selected.append(x[:, -1] + attention[:, -1])
    return torch.stack(selected), torch.stack(updates), state[:, -1]


def truncated(model, state, kwargs, end=20, start=17):
    """Run only blocks start..end; later blocks and the output head are not used."""
    validate(model, start, (end,))
    x = state
    for layer in model.model.layers[start : end + 1]:
        x = layer(x, **kwargs)
        x = x[0] if isinstance(x, tuple) else x
    return x[:, -1]


def accumulation_jvp(model, state, kwargs, delta, start=17, exits=(20, 23, 26)):
    """Propagate a frozen initial direction through all exits in one forward JVP."""
    with torch.no_grad():
        primal, tangent = torch.func.jvp(
            lambda h: accumulate(model, h, kwargs, start, exits), (state,), (delta,)
        )
    return tuple(x.detach() for x in primal), tuple(x.detach() for x in tangent)


def truncated_jvp(model, state, kwargs, delta, end=20, start=17):
    """Verify a real early-exit executable, not cached output from a full traversal."""
    with torch.no_grad():
        _, tangent = torch.func.jvp(
            lambda h: truncated(model, h, kwargs, end, start), (state,), (delta,)
        )
    return tangent.detach()
