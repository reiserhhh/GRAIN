#!/usr/bin/env python3
"""Full query/key/value directional attention responses at a fixed boundary."""

import numpy as np

from .fixed_seed_attention import rotary

CHANNELS = ("query", "key", "value")


def pre_attention(model, state, kwargs, start=17, block=22):
    """Return the full token state before the selected block, without running it."""
    if model.config.model_type != "qwen2" or not 0 <= start <= block < len(
        model.model.layers
    ):
        raise ValueError("Pinned Qwen2 block range required")
    x = state
    for layer in model.model.layers[start:block]:
        x = layer(x, **kwargs)
        x = x[0] if isinstance(x, tuple) else x
    return x


def export_weights(model, block=22):
    """Export actual attention projections, biases and input RMSNorm parameters."""
    layer = model.model.layers[block]
    a = layer.self_attn
    if (
        model.config.model_type != "qwen2"
        or model.training
        or a.sliding_window is not None
        or model.config._attn_implementation != "eager"
    ):
        raise ValueError("Pinned eager/eval/full-attention Qwen2 required")
    out = {}
    for name in ("q", "k", "v", "o"):
        p = getattr(a, name + "_proj")
        out[name] = p.weight.detach().cpu().numpy().copy()
        out[name + "_bias"] = (
            np.zeros(p.out_features, np.float32)
            if p.bias is None
            else p.bias.detach().cpu().numpy().copy()
        )
    out.update(
        gamma=layer.input_layernorm.weight.detach().cpu().numpy().copy(),
        epsilon=np.asarray(layer.input_layernorm.variance_epsilon),
        head_dim=np.asarray(a.head_dim),
        scale=np.asarray(a.scaling),
        block=np.asarray(block),
    )
    return out


def normalized_directions(x, d, gamma, epsilon):
    """Apply RMSNorm and its exact radial derivative independently at every token."""
    x, d, gamma = [np.asarray(v, float) for v in (x, d, gamma)]
    if (
        x.ndim != 2
        or d.ndim != 3
        or d.shape[1:] != x.shape
        or min(d.shape) < 1
        or gamma.shape != (x.shape[1],)
        or not np.isfinite(epsilon)
        or epsilon <= 0
        or not all(np.isfinite(v).all() for v in (x, d, gamma))
    ):
        raise ValueError(
            "Finite full token states/directions and positive RMS epsilon required"
        )
    rho = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon)
    z = gamma * x / rho
    dz = gamma * (d / rho - np.mean(d * x, axis=-1, keepdims=True) * x / rho**3)
    return z, dz


def decompose(x, d, cos, sin, mask, weights):
    """Return all three final-position channels, retaining every attention head.

    x is [tokens, hidden]; d is [directions, tokens, hidden]. The fixed additive
    mask is the last query's admissibility row. No residual identity is included.
    """
    w = {k: np.asarray(v, float) for k, v in weights.items()}
    z, dz = normalized_directions(x, d, w["gamma"], float(w["epsilon"]))
    tokens, hidden = z.shape
    hd = int(w["head_dim"])
    cos, sin, mask = [np.asarray(v, float) for v in (cos, sin, mask)]
    if (
        hd <= 0
        or hd % 2
        or any(w[n].ndim != 2 or w[n].shape[1] != hidden for n in ("q", "k", "v"))
        or w["k"].shape != w["v"].shape
        or any(w[n].shape[0] % hd for n in ("q", "k"))
        or w["o"].shape != (hidden, w["q"].shape[0])
        or any(w[n + "_bias"].shape != (w[n].shape[0],) for n in ("q", "k", "v", "o"))
        or cos.shape != (tokens, hd)
        or sin.shape != cos.shape
        or mask.shape != (tokens,)
        or not np.isfinite(cos).all()
        or not np.isfinite(sin).all()
        or np.isnan(mask).any()
        or np.isposinf(mask).any()
        or not np.isfinite(mask).any()
        or float(w["scale"]) <= 0
        or not all(np.isfinite(v).all() for v in w.values())
    ):
        raise ValueError(
            "Aligned finite projections, rotary positions and mask required"
        )
    heads, kv_heads = w["q"].shape[0] // hd, w["k"].shape[0] // hd
    if kv_heads < 1 or heads < 1 or heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    factors, groups = len(dz), heads // kv_heads
    q = (z[-1] @ w["q"].T + w["q_bias"]).reshape(heads, hd)
    k = (z @ w["k"].T + w["k_bias"]).reshape(tokens, kv_heads, hd)
    v = (z @ w["v"].T + w["v_bias"]).reshape(tokens, kv_heads, hd)
    dq = (dz[:, -1] @ w["q"].T).reshape(factors, heads, hd)
    dk = (dz @ w["k"].T).reshape(factors, tokens, kv_heads, hd)
    dv = (dz @ w["v"].T).reshape(factors, tokens, kv_heads, hd)
    q, dq = rotary(q, cos[-1], sin[-1]), rotary(dq, cos[-1], sin[-1])
    k = rotary(k, cos[:, None], sin[:, None])
    dk = rotary(dk, cos[None, :, None], sin[None, :, None])
    parts = np.zeros((3, factors, heads, hd))
    outputs, masses = [], []
    scale = float(w["scale"])
    for h in range(heads):
        kv = h // groups
        scores = k[:, kv] @ q[h] * scale + mask
        a = np.exp(scores - np.max(scores))
        a /= a.sum()
        o = a @ v[:, kv]
        centered = v[:, kv] - o
        ds_q = dq[:, h] @ k[:, kv].T * scale
        ds_k = np.einsum("fth,h->ft", dk[:, :, kv], q[h]) * scale
        parts[0, :, h] = (ds_q * a) @ centered
        parts[1, :, h] = (ds_k * a) @ centered
        parts[2, :, h] = np.einsum("t,fth->fh", a, dv[:, :, kv])
        outputs.append(o)
        masses.append(a)
    projected = parts.reshape(3, factors, -1) @ w["o"].T
    return {
        "channels": projected,
        "head_channels": parts,
        "primal": np.stack(outputs).reshape(-1) @ w["o"].T + w["o_bias"],
        "attention": np.stack(masses),
    }
