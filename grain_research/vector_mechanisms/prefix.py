#!/usr/bin/env python3
"""Frozen Qwen prefix capture, terminating before suffix attention and scoring."""

from __future__ import annotations

import numpy as np
import torch

from grain_research.vector_mechanisms.query_tilt import (
    pairing_component,
    tilted_readout,
)


class _PrefixReady(Exception):
    """Private control-flow signal; never intercept unrelated model exceptions."""


def rotate_half(x):
    """Rotate paired half coordinates in the Qwen rotary convention."""
    if x.shape[-1] % 2:
        raise ValueError("Rotary dimension must be even")
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def rotate_query(raw, cos, sin, scaling):
    """Place a raw head query at the current endpoint position, then apply scale."""
    raw, cos, sin = (np.asarray(a) for a in (raw, cos, sin))
    if raw.ndim != 2 or cos.shape != (raw.shape[1],) or sin.shape != cos.shape:
        raise ValueError("Invalid rotary query shapes")
    return (raw * cos + rotate_half(raw) * sin) * scaling


def capture_prefix(observer, ids, block=16, layer=17):
    """Capture only a causal prefix and the next layer's QKV; no suffix/logit call.

    All hooks are scoped, and the original module graph and weights are retained.
    The returned trace proves which blocks completed. Full weights remain resident.
    """
    model = observer.model
    layers = model.model.layers
    if layer != block + 1 or not 0 <= block < layer < len(layers):
        raise ValueError("Consecutive valid prefix boundary and QKV layer required")
    if len(ids) > observer.max_tokens or not len(ids):
        raise ValueError("Invalid prefix token length")
    attention = layers[layer].self_attn
    if attention.sliding_window is not None:
        raise ValueError("Fixed prefix experiment requires unwindowed attention")
    saved, handles, trace = {}, [], {"completed_blocks": [], "forbidden_calls": []}

    def completed(index):
        def remember(_module, _args, output):
            trace["completed_blocks"].append(index)
            if index == block:
                h = output[0] if isinstance(output, tuple) else output
                saved["boundary"] = h[0].detach().float().cpu().numpy().copy()

        return remember

    def forbidden(name):
        def reject(_module, _args):
            trace["forbidden_calls"].append(name)
            raise RuntimeError("Prefix executed forbidden module: " + name)

        return reject

    def attention_input(_module, _args, kwargs):
        cos, sin = kwargs["position_embeddings"]
        saved["cos"] = cos[0].detach().float().cpu().numpy().copy()
        saved["sin"] = sin[0].detach().float().cpu().numpy().copy()
        mask = kwargs.get("attention_mask")
        if mask is not None and torch.any(mask[0, 0, -1] != 0):
            raise ValueError("Final query must attend to full compact prefix")

    def projection(name):
        def capture(_module, _inputs, output):
            tensor = output[0, -1] if name == "q" else output[0]
            saved[name] = tensor.detach().float().cpu().numpy().copy()
            if name == "v":
                raise _PrefixReady

        return capture

    try:
        for index in range(block + 1):
            handles.append(layers[index].register_forward_hook(completed(index)))
        for index in range(layer + 1, len(layers)):
            handles.append(
                layers[index].register_forward_pre_hook(forbidden(f"block_{index}"))
            )
        for name, module in (
            ("layer_attention_output", attention.o_proj),
            ("layer_mlp", layers[layer].mlp),
            ("final_norm", model.model.norm),
            ("lm_head", model.lm_head),
        ):
            handles.append(module.register_forward_pre_hook(forbidden(name)))
        handles.append(
            attention.register_forward_pre_hook(attention_input, with_kwargs=True)
        )
        for name in ("q", "k", "v"):
            handles.append(
                getattr(attention, name + "_proj").register_forward_hook(
                    projection(name)
                )
            )
        tokens = torch.tensor([ids], device=next(model.parameters()).device)
        with torch.no_grad():
            try:
                model.model(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                    return_dict=True,
                )
            except _PrefixReady:
                pass
            else:
                raise RuntimeError("Prefix did not terminate at the QKV site")
    finally:
        for handle in handles:
            handle.remove()
    observer.calls += 1
    if trace["completed_blocks"] != list(range(block + 1)) or trace["forbidden_calls"]:
        raise RuntimeError("Unexpected prefix execution trace")
    dim = attention.head_dim
    raw = saved["q"].reshape(-1, dim)
    k = saved["k"].reshape(len(ids), -1, dim)
    v = saved["v"].reshape(len(ids), -1, dim)
    cos, sin = saved["cos"], saved["sin"]
    return {
        "boundary": saved["boundary"],
        "raw_query": raw,
        "q": rotate_query(raw, cos[-1], sin[-1], attention.scaling),
        "k": k * cos[:, None] + rotate_half(k) * sin[:, None],
        "v": v,
        "last_cos": cos[-1],
        "last_sin": sin[-1],
        "scaling": attention.scaling,
    }, trace


def prefix_pairing(cache, groups, queries=None):
    """Compute selected-comment pairing for each head from captured prefix arrays."""
    q = cache["q"] if queries is None else np.asarray(queries)
    k, v = cache["k"], cache["v"]
    if q.shape != cache["q"].shape or len(q) % k.shape[1]:
        raise ValueError("Incompatible head queries")
    ratio, parts = len(q) // k.shape[1], []
    for head, query in enumerate(q):
        _, p = tilted_readout(k[:, head // ratio], v[:, head // ratio], query)
        parts.append(pairing_component(p, v[:, head // ratio], groups))
    return np.concatenate(parts)


def affine_readout(views, model_path):
    """Average fixed view vectors and replay one frozen affine author reader."""
    x = np.asarray(views, float)
    with np.load(model_path, allow_pickle=False) as model:
        coef, intercept = model["coef"], float(model["intercept"])
    if x.ndim != 2 or x.shape[1:] != coef.shape or not np.isfinite(x).all():
        raise ValueError("View vectors do not match frozen affine reader")
    return float(x.mean(0) @ coef + intercept)
