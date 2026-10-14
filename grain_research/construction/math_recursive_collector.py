#!/usr/bin/env python3
"""Layer-indexed independent context libraries and an explicit current recurrence."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np


def stream_sha(path):
    """Authenticate large frozen weights without materializing their file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parity_laws():
    """Return positive laws with equal degree0--2 moments and different degree3."""
    signs = np.array(list(itertools.product((-1, 1), repeat=3)), int)
    chi = np.prod(signs, axis=1)
    counts = np.stack((np.where(chi == 1, 7, 3), np.where(chi == 1, 3, 7)))
    return signs, counts


def parity_account(values):
    """Integrate all qualified corners; expose the exact third-character action."""
    f = np.asarray(values, float)
    signs, counts = parity_laws()
    if f.ndim != 2 or len(f) != 8 or not np.isfinite(f).all():
        raise ValueError("Eight finite qualified corners required")
    coefficient = (np.prod(signs, axis=1)[:, None] * f).mean(0)
    scores = counts @ f / counts.sum(1, keepdims=True)
    return scores, coefficient


def rms_norm(h, weight, eps):
    """Apply stock float32 tokenwise RMS normalization, not cohort normalization."""
    import torch

    x = h.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(h.dtype)


def rotary(x, cos, sin):
    """Rotate token-major Q/K coordinates with the fixed Qwen half convention."""
    import torch

    if x.shape[-1] % 2 or cos.shape != sin.shape:
        raise ValueError("Compatible even rotary coordinates required")
    d = x.shape[-1] // 2
    half = torch.cat((-x[..., d:], x[..., :d]), dim=-1)
    return x * cos[..., None, :] + half * sin[..., None, :]


def gqa_attention(q, k, v, allowed):
    """Compute current-query GQA from token-major arrays and explicit support."""
    import torch

    if (
        k.shape != v.shape
        or q.ndim != 4
        or k.ndim != 4
        or q.shape[2] % k.shape[2]
        or allowed.shape != (q.shape[0], q.shape[1], k.shape[1])
        or not bool(allowed.any(-1).all())
    ):
        raise ValueError("Complete compatible GQA/support required")
    # Repeat each KV head consecutively, as in Qwen repeat_kv.
    keys = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2).transpose(1, 2)
    values = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2).transpose(1, 2)
    logits = q.transpose(1, 2) @ keys.transpose(-1, -2) * q.shape[-1] ** -0.5
    logits = logits.masked_fill(~allowed[:, None], torch.finfo(logits.dtype).min)
    p = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
    return (p @ values).transpose(1, 2).contiguous()


def current_block(h, context_k, context_v, allowed, cos, sin, weights, config):
    """Execute one declared decoder block only on current tokens, without modules."""
    import torch
    from torch.nn import functional as f

    batch, length, hidden = h.shape
    kv = config["num_key_value_heads"]
    dim = hidden // config["num_attention_heads"]
    context_length = context_k.shape[1]
    x = rms_norm(h, weights["input_layernorm.weight"], config["rms_norm_eps"])

    def project(name, heads):
        """Apply the immutable affine map in its native output-head ordering."""
        return f.linear(
            x,
            weights[f"self_attn.{name}_proj.weight"],
            weights[f"self_attn.{name}_proj.bias"],
        ).reshape(batch, length, heads, dim)

    q = rotary(
        project("q", config["num_attention_heads"]),
        cos[context_length:],
        sin[context_length:],
    )
    k = rotary(project("k", kv), cos[context_length:], sin[context_length:])
    v = project("v", kv)
    history_k = rotary(context_k, cos[:context_length], sin[:context_length])
    attended = gqa_attention(
        q, torch.cat((history_k, k), dim=1), torch.cat((context_v, v), dim=1), allowed
    )
    h = h + f.linear(
        attended.reshape(batch, length, hidden), weights["self_attn.o_proj.weight"]
    )
    x = rms_norm(h, weights["post_attention_layernorm.weight"], config["rms_norm_eps"])
    gate = f.silu(f.linear(x, weights["mlp.gate_proj.weight"]))
    update = f.linear(
        gate * f.linear(x, weights["mlp.up_proj.weight"]),
        weights["mlp.down_proj.weight"],
    )
    return h + update


def assemble_context(library, cases, layer, context_length):
    """Assemble independent P/E/R donors, never using observed joint states."""
    keys, values, masks = [], [], []
    for case in cases:
        if (
            len(case["levels"]) != 6
            or any(level not in (0, 1, 2) for level in case["levels"])
            or case["owner"] not in (-1, 1)
        ):
            raise ValueError("Six fixed event levels and an explicit owner required")
        shape = (context_length, *library["p_k"].shape[2:])
        k, v, valid = (
            np.zeros(shape, np.float32),
            np.zeros(shape, np.float32),
            np.zeros(context_length, bool),
        )
        names = [
            "p",
            *(f"e{j}_l{level}" for j, level in enumerate(case["levels"])),
            f"r_o{case['owner']}",
        ]
        for name in names:
            pos = library[name + "_pos"]
            if valid[pos].any() or np.any(pos >= context_length):
                raise ValueError("Overlapping or non-context library addresses")
            k[pos], v[pos], valid[pos] = (
                library[name + "_k"][layer],
                library[name + "_v"][layer],
                True,
            )
        keys.append(k)
        values.append(v)
        masks.append(valid)
    return np.stack(keys), np.stack(values), np.stack(masks)


def load_weights(path, blocks, device):
    """Load only named frozen decoder weights; never instantiate a language model."""
    import torch
    from safetensors import safe_open

    names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    final = [
        "input_layernorm.weight",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
    ]
    with safe_open(str(path), framework="pt", device="cpu") as source:
        result = [
            {
                name: source.get_tensor(f"model.layers.{layer}.{name}").to(
                    device=device, dtype=torch.float32
                )
                for name in (names if layer < blocks else final)
            }
            for layer in range(blocks + 1)
        ]
    return result


def compile_current(library, cases, weights, config, variant, device):
    """Run the fixed layer recurrence on numeric addresses with no tokenizer/model."""
    import torch
    from torch.nn import functional as f

    if variant not in ("layered", "broadcast_last_history") or not cases:
        raise ValueError("Declared finite variant and cases required")
    blocks = len(weights) - 1
    seed = torch.as_tensor(library["seed"], device=device)
    cos, sin = (
        torch.as_tensor(library[name], device=device) for name in ("cos", "sin")
    )
    h = seed[None].expand(len(cases), -1, -1).clone()
    context_length = len(cos) - len(seed)
    for layer in range(blocks):
        donor_layer = layer if variant == "layered" else blocks - 1
        k, v, valid = assemble_context(library, cases, donor_layer, context_length)
        current_support = np.tri(len(seed), dtype=bool)
        allowed = np.concatenate(
            (
                np.broadcast_to(
                    valid[:, None], (len(cases), len(seed), context_length)
                ),
                np.broadcast_to(current_support, (len(cases), len(seed), len(seed))),
            ),
            axis=-1,
        ).copy()
        h = current_block(
            h,
            torch.as_tensor(k, device=device),
            torch.as_tensor(v, device=device),
            torch.as_tensor(allowed, device=device),
            cos,
            sin,
            weights[layer],
            config,
        )
    x = rms_norm(h, weights[-1]["input_layernorm.weight"], config["rms_norm_eps"])
    dim = config["hidden_size"] // config["num_attention_heads"]
    outputs = []
    for name in ("k", "v"):
        raw = f.linear(
            x,
            weights[-1][f"self_attn.{name}_proj.weight"],
            weights[-1][f"self_attn.{name}_proj.bias"],
        )
        outputs.append(
            raw.reshape(len(cases), len(seed), config["num_key_value_heads"], dim)
            .detach()
            .cpu()
            .numpy()
        )
    # HM111's final captured key applies this NumPy rotary convention.
    k, v = outputs
    d = k.shape[-1] // 2
    half = np.concatenate((-k[..., d:], k[..., :d]), axis=-1)
    k = (
        k * library["cos"][context_length:, None]
        + half * library["sin"][context_length:, None]
    )
    if not np.isfinite(k).all() or not np.isfinite(v).all():
        raise FloatingPointError("Nonfinite constructed current")
    return k, v
