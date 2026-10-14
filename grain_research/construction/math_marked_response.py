#!/usr/bin/env python3
"""Availability-aware path measures with frozen, anonymous GRAIN marks.

The paths interpolate stored layer histories, not natural texts. All pretrained
maps are retained; this is an instrument response, not a human causal effect.
"""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_recursive_collector as C


def gated_attention(q, k, v, availability):
    """Extend masked GQA to nonnegative key mass, including exact zero mass."""
    import torch

    ratio = q.shape[2] // k.shape[2]
    if (
        k.shape != v.shape
        or q.shape[2] % k.shape[2]
        or availability.shape != (q.shape[0], q.shape[1], k.shape[1])
        or bool((availability < 0).any())
        or not bool((availability.sum(-1) > 0).all())
    ):
        raise ValueError("Nonnegative nonempty key mass and compatible GQA required")
    keys = k.repeat_interleave(ratio, dim=2).transpose(1, 2)
    values = v.repeat_interleave(ratio, dim=2).transpose(1, 2)
    logits = q.transpose(1, 2) @ keys.transpose(-1, -2) * q.shape[-1] ** -0.5
    # Masked keys must not set the stabilization origin. At interior points all
    # varying-mass keys are positive; excluded keys have zero path derivative.
    origin = (
        logits.masked_fill(availability[:, None] == 0, -torch.inf)
        .amax(-1, keepdim=True)
        .detach()
    )
    mass = availability[:, None] * torch.exp(logits - origin)
    probability = mass / mass.sum(-1, keepdim=True)
    return (probability @ values).transpose(1, 2).contiguous()


def block(h, history_k, history_v, gate, cos, sin, weights, config):
    """Run one stock current block with a declared history-key mass extension."""
    import torch
    from torch.nn import functional as f

    batch, length, hidden = h.shape
    context, kv = history_k.shape[1], config["num_key_value_heads"]
    heads, dim = config["num_attention_heads"], hidden // config["num_attention_heads"]
    x = C.rms_norm(h, weights["input_layernorm.weight"], config["rms_norm_eps"])

    def project(name, count):
        return f.linear(
            x,
            weights[f"self_attn.{name}_proj.weight"],
            weights[f"self_attn.{name}_proj.bias"],
        ).reshape(batch, length, count, dim)

    q = C.rotary(project("q", heads), cos[context:], sin[context:])
    k = C.rotary(project("k", kv), cos[context:], sin[context:])
    v = project("v", kv)
    hk = C.rotary(history_k, cos[:context], sin[:context])
    causal = torch.ones(length, length, device=h.device, dtype=h.dtype).tril()
    support = torch.cat(
        (gate[:, None].expand(-1, length, -1), causal[None].expand(batch, -1, -1)),
        dim=-1,
    )
    attention = gated_attention(
        q, torch.cat((hk, k), 1), torch.cat((history_v, v), 1), support
    )
    h = h + f.linear(
        attention.reshape(batch, length, hidden), weights["self_attn.o_proj.weight"]
    )
    x = C.rms_norm(
        h, weights["post_attention_layernorm.weight"], config["rms_norm_eps"]
    )
    return h + f.linear(
        f.silu(f.linear(x, weights["mlp.gate_proj.weight"]))
        * f.linear(x, weights["mlp.up_proj.weight"]),
        weights["mlp.down_proj.weight"],
    )


def construct(k, v, gate, seeds, cos, sin, weights, config):
    """Differentiate current states with respect to layer K/V and availability."""
    from torch.nn import functional as f

    batch, focuses, length = k.shape[0], seeds.shape[0], seeds.shape[1]
    h = seeds[None].expand(batch, -1, -1, -1).reshape(batch * focuses, length, -1)
    for layer in range(len(weights) - 1):
        h = block(
            h,
            k[:, layer].repeat_interleave(focuses, dim=0),
            v[:, layer].repeat_interleave(focuses, dim=0),
            gate.repeat_interleave(focuses, dim=0),
            cos,
            sin,
            weights[layer],
            config,
        )
    x = C.rms_norm(h, weights[-1]["input_layernorm.weight"], config["rms_norm_eps"])
    dim = config["hidden_size"] // config["num_attention_heads"]
    result = []
    for name in ("k", "v"):
        a = f.linear(
            x,
            weights[-1][f"self_attn.{name}_proj.weight"],
            weights[-1][f"self_attn.{name}_proj.bias"],
        ).reshape(batch * focuses, length, config["num_key_value_heads"], dim)
        if name == "k":
            a = C.rotary(a, cos[k.shape[2] :], sin[k.shape[2] :])
        result.append(a.reshape(batch, focuses, length, -1, dim))
    return tuple(result)


def read(k, v, queries, directions):
    """Return batch/focus/mode/target values in the inherited reader units."""
    import torch

    ratio = directions.shape[1] // k.shape[-2]
    k, v = (a.repeat_interleave(ratio, dim=-2) for a in (k, v))
    logits = torch.einsum("brihd,qhd->brqhi", k, queries)
    content = torch.einsum("brihd,qhd->brqhi", v, directions)
    uniform = content.mean(-1).sum(-1)
    weighted = (torch.softmax(logits, -1) * content).sum((-1, -2))
    return torch.stack((uniform, weighted), dim=2)


def quadrature(nodes=4):
    """Use one fixed Gauss-Legendre rule; no result-selected refinement."""
    if nodes != 4:
        raise ValueError("This finite protocol uses exactly four nodes")
    x, w = np.polynomial.legendre.leggauss(nodes)
    return (x + 1) / 2, w / 2


def marked_moments(attributions, marks, visible):
    """Keep signed mass, first moment and within-evidence connected moment.

    Marks are endpoint-average anonymous coordinates on matching token slots.
    They are not a latent natural-text coordinate along the interpolated path.
    Untagged prompt slots retain their mass instead of acquiring invented marks.
    """
    a, g, mask = (
        np.asarray(attributions, float),
        np.asarray(marks, float),
        np.asarray(visible, bool),
    )
    if (
        a.ndim != 4
        or g.shape != (a.shape[0], a.shape[-1], 4)
        or mask.shape != g.shape[:2]
    ):
        raise ValueError(
            "Pair/channel/target/token mass and four endpoint marks required"
        )
    if not np.isfinite(a).all() or not np.isfinite(g).all() or not mask.any(1).all():
        raise ValueError("Finite mass and nonempty declared evidence marks required")
    average = (g * mask[..., None]).sum(1) / mask.sum(1)[:, None]
    tagged = a * mask[:, None, None]
    first = np.einsum("pcqi,pij->pcqj", tagged, g)
    connected = first - tagged.sum(-1)[..., None] * average[:, None, None]
    shifted = np.zeros_like(g)
    for p in range(len(g)):
        indices = np.flatnonzero(mask[p])
        shifted[p, indices] = np.roll(g[p, indices], len(indices) // 2, axis=0)
    shuffled = np.einsum("pcqi,pij->pcqj", tagged, shifted - average[:, None])
    return {
        "mass": a.sum(-1),
        "body_mass": tagged.sum(-1),
        "untagged_mass": (a - tagged).sum(-1),
        "first": first,
        "connected": connected,
        "misaligned_connected": shuffled,
        "constant_connected": np.zeros_like(connected),
        "total_variation": abs(a).sum(-1),
    }
