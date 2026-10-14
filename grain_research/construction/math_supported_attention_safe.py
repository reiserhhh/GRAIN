#!/usr/bin/env python3
"""Numerically defined supported-measure attention; zero mass has no exponent."""

from grain_research.construction import math_recursive_collector as C
from grain_research.construction.math_supported_measure import moments as moments
from grain_research.construction.math_supported_measure import pack as pack

OVERFLOW_WITNESSES = []


def attention(q, k, v, mass):
    """Use log mass plus logits; only positive-mass atoms enter normalization."""
    import torch

    ratio = q.shape[2] // k.shape[2]
    keys = k.repeat_interleave(ratio, 2).transpose(1, 2)
    values = v.repeat_interleave(ratio, 2).transpose(1, 2)
    logits = q.transpose(1, 2) @ keys.transpose(-1, -2) * q.shape[-1] ** -0.5
    active = mass[:, None] > 0
    if not OVERFLOW_WITNESSES and bool((~active).any()):
        origin = logits.masked_fill(~active, -torch.inf).amax(-1, keepdim=True)
        extreme = (
            (logits - origin).masked_fill(active, -torch.inf).amax().detach().item()
        )
        if extreme > 88.72:
            OVERFLOW_WITNESSES.append(
                {"inactive_exponent_max": extreme, "float32_log_max": 88.72}
            )
    positive = torch.where(active, mass[:, None], 1)
    log_mass = positive.log().masked_fill(~active, -torch.inf)
    probability = torch.softmax(logits + log_mass, dim=-1)
    return (probability @ values).transpose(1, 2).contiguous()


def block(h, hk, hv, mass, hc, hs, cc, cs, weights, cfg):
    """Stock current block with declared supported history measure and positions."""
    import torch
    from torch.nn import functional as f

    b, length, hidden = h.shape
    heads, kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    dim = hidden // heads
    x = C.rms_norm(h, weights["input_layernorm.weight"], cfg["rms_norm_eps"])

    def project(name, count):
        """Use the immutable affine attention maps in native head ordering."""
        return f.linear(
            x,
            weights[f"self_attn.{name}_proj.weight"],
            weights[f"self_attn.{name}_proj.bias"],
        ).reshape(b, length, count, dim)

    q, k = C.rotary(project("q", heads), cc, cs), C.rotary(project("k", kv), cc, cs)
    v = project("v", kv)
    hk = C.rotary(hk, hc, hs)
    causal = torch.ones(length, length, device=h.device, dtype=h.dtype).tril()
    support = torch.cat(
        (mass[:, None].expand(-1, length, -1), causal[None].expand(b, -1, -1)), dim=-1
    )
    attended = attention(q, torch.cat((hk, k), 1), torch.cat((hv, v), 1), support)
    h = h + f.linear(
        attended.reshape(b, length, hidden), weights["self_attn.o_proj.weight"]
    )
    x = C.rms_norm(h, weights["post_attention_layernorm.weight"], cfg["rms_norm_eps"])
    return h + f.linear(
        f.silu(f.linear(x, weights["mlp.gate_proj.weight"]))
        * f.linear(x, weights["mlp.up_proj.weight"]),
        weights["mlp.down_proj.weight"],
    )


def construct(k, v, mass, positions, current_origin, seeds, cos, sin, weights, config):
    """Execute the same measure path without evaluating invisible exponents."""
    import torch
    from torch.nn import functional as f

    positions = torch.as_tensor(positions, dtype=torch.long, device=cos.device)
    if bool((positions < 0).any()) or bool((positions >= current_origin).any()):
        raise ValueError("Old atom positions must precede current origin")
    batch, focuses, length = k.shape[0], seeds.shape[0], seeds.shape[1]
    h = seeds[None].expand(batch, -1, -1, -1).reshape(batch * focuses, length, -1)
    for layer in range(len(weights) - 1):
        h = block(
            h,
            k[:, layer].repeat_interleave(focuses, 0),
            v[:, layer].repeat_interleave(focuses, 0),
            mass.repeat_interleave(focuses, 0),
            cos[positions],
            sin[positions],
            cos[current_origin:],
            sin[current_origin:],
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
            a = C.rotary(a, cos[current_origin:], sin[current_origin:])
        if not bool(torch.isfinite(a).all()):
            raise FloatingPointError("Supported constructor remains nonfinite")
        result.append(a.reshape(batch, focuses, length, -1, dim))
    return tuple(result)
