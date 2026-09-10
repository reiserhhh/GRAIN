#!/usr/bin/env python3
"""Exact SwiGLU/RMS directional updates and source-frozen local alternatives."""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import expit

ARMS = ("current", "source_B", "source_N", "source_BN")


def pre_mlp(model, state, kwargs, start=17, block=25):
    """Stop immediately after the selected attention; never run its MLP or later blocks."""
    if model.config.model_type != "qwen2" or not 0 <= start <= block < len(
        model.model.layers
    ):
        raise ValueError("Pinned Qwen2 block range required")
    x = state
    for layer in model.model.layers[start:block]:
        x = layer(x, **kwargs)
        x = x[0] if isinstance(x, tuple) else x
    layer = model.model.layers[block]
    a, _ = layer.self_attn(layer.input_layernorm(x), **kwargs)
    return (x + a)[:, -1]


def capture_jvp(model, state, kwargs, direction, start=17, block=25):
    """Collect the current input and incoming tangent without any score/backward."""
    with torch.no_grad():
        x, d = torch.func.jvp(
            lambda h: pre_mlp(model, h, kwargs, start, block), (state,), (direction,)
        )
    return x.detach(), d.detach()


def export_weights(model, block=25):
    """Extract the pinned bias-free SiLU weights and RMS parameters without fitting."""
    layer = model.model.layers[block]
    if model.config.model_type != "qwen2" or model.config.hidden_act != "silu":
        raise ValueError("Actual bias-free Qwen2 SiLU architecture required")
    parts = {
        "gate": layer.mlp.gate_proj,
        "up": layer.mlp.up_proj,
        "down": layer.mlp.down_proj,
    }
    if any(m.bias is not None for m in parts.values()):
        raise ValueError("Unexpected projection bias")
    result = {k: m.weight.detach().cpu().numpy().copy() for k, m in parts.items()}
    result["gamma"] = (
        layer.post_attention_layernorm.weight.detach().cpu().numpy().copy()
    )
    result["epsilon"] = np.asarray(layer.post_attention_layernorm.variance_epsilon)
    return result


def rms_direction(x, d, weights):
    """Apply RMSNorm's full derivative, including the radial subtraction and epsilon."""
    x, d = np.asarray(x, dtype=float), np.asarray(d, dtype=float)
    rho = np.sqrt(np.mean(x * x) + float(weights["epsilon"]))
    if rho <= 0 or d.shape[-1] != x.size or x.ndim != 1:
        raise ValueError("Finite positive RMS and matching vectors required")
    return weights["gamma"] * (
        d / rho - np.sum(d * x, axis=-1, keepdims=True) * x / (x.size * rho**3)
    )


def coefficients(x, weights):
    """Return the two diagonal SwiGLU differential coefficients at a fixed context."""
    x = np.asarray(x, dtype=float)
    z = x * weights["gamma"] / np.sqrt(np.mean(x * x) + float(weights["epsilon"]))
    g, u = weights["gate"] @ z, weights["up"] @ z
    sigmoid = expit(g)
    return np.stack((g * sigmoid, u * sigmoid * (1 + g * (1 - sigmoid))))


def apply_coefficients(coeff, dz, weights):
    """Keep input-variation and gate-variation vectors separate before summation."""
    input_term = (dz @ weights["up"].T * coeff[0]) @ weights["down"].T
    gate_term = (dz @ weights["gate"].T * coeff[1]) @ weights["down"].T
    return np.stack((input_term, gate_term))


def source_members(indices, queries, query, held=None):
    """Choose only same-query source authors and exclude the held source author."""
    indices, queries = np.asarray(indices), np.asarray(queries)
    if indices.shape != queries.shape or indices.ndim != 1:
        raise ValueError("Source membership arrays must match")
    if len(set(zip(indices.tolist(), queries.tolist(), strict=True))) != len(indices):
        raise ValueError("Duplicate source author/query")
    keep = (queries == query) & (indices != held if held is not None else True)
    selected = np.flatnonzero(keep)
    if not len(selected):
        raise ValueError("No independent same-query source")
    return selected


def alternatives(x, d, source_x, weights):
    """Evaluate fixed current/source B and N choices without changing incoming d.

    The paired arm averages B_s N_s, not the generally different product of means.
    All returned arrays have shape (2 terms, directions, hidden width).
    """
    x, d, source_x = (np.asarray(a, dtype=float) for a in (x, d, source_x))
    if source_x.ndim != 2 or not len(source_x) or source_x.shape[1] != x.size:
        raise ValueError("Nonempty source vectors of the same width required")
    c = coefficients(x, weights)
    n = rms_direction(x, d, weights)
    source_c = np.stack([coefficients(s, weights) for s in source_x])
    source_n = np.stack([rms_direction(s, d, weights) for s in source_x])
    return {
        "current": apply_coefficients(c, n, weights),
        "source_B": apply_coefficients(source_c.mean(0), n, weights),
        "source_N": apply_coefficients(c, source_n.mean(0), weights),
        "source_BN": np.mean(
            [
                apply_coefficients(a, b, weights)
                for a, b in zip(source_c, source_n, strict=True)
            ],
            axis=0,
        ),
    }
