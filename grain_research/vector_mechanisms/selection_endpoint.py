#!/usr/bin/env python3
"""Full key-selection increments and their declared downstream endpoint."""

from __future__ import annotations

import numpy as np

from .compiled_query_transfer import QUERIES, validate_sources
from .fixed_seed_attention import rotary
from .source_query_selection import FIELDS, key_actions


def key_vectors(query_anchors, evidence, output_matrix):
    """Return [anchor,factor,hidden] increments without a current query/readout.

    Reuse NEXT50's validation, but compute the vector before any scalar readout.
    The two zero readouts validate shapes only and do not affect this calculation.
    """
    if set(evidence) != FIELDS - {"g"}:
        raise ValueError("Pure K/V/dK evidence required; no current Q/readout")
    q = np.asarray(query_anchors, float)
    o = np.asarray(output_matrix, float)
    if q.ndim != 3 or o.ndim != 2 or o.shape[1] != q.shape[1] * q.shape[2]:
        raise ValueError("Aligned heads and output matrix required")
    if not np.isfinite(o).all():
        raise ValueError("Finite output matrix required")
    dummy = evidence | {"g": np.zeros((2, q.shape[1], q.shape[2]))}
    key_actions(q, dummy)
    e = {k: np.asarray(v, float) for k, v in evidence.items()}
    q = rotary(q, e["cos"][-1], e["sin"][-1])
    k = rotary(e["k"], e["cos"][:, None], e["sin"][:, None])
    dk = rotary(e["dk"], e["cos"][None, :, None], e["sin"][None, :, None])
    result = np.zeros((len(q), len(dk), q.shape[1], q.shape[2]))
    groups = q.shape[1] // k.shape[1]
    for a in range(len(q)):
        for h in range(q.shape[1]):
            kv = h // groups
            logits = k[:, kv] @ q[a, h] * float(e["scale"]) + e["mask"]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            centered = e["v"][:, kv] - p @ e["v"][:, kv]
            delta = np.einsum("fth,h->ft", dk[:, :, kv], q[a, h]) * float(e["scale"])
            result[a, :, h] = (delta * p) @ centered
    return result.reshape(len(q), len(dk), -1) @ o.T


def source_reader(bank, cohort, index, query):
    """Average only permitted same-query source suffix gradients."""
    if set(bank) != {(i, q) for i in range(4) for q in QUERIES}:
        raise ValueError("Exact source4 paired-query bank required")
    if query not in QUERIES:
        raise ValueError("Known query required")
    members = [i for i in range(4) if cohort != "source4" or i != index]
    validate_sources(cohort, index, members)
    rows = [np.asarray(bank[i, query], float) for i in members]
    if any(
        x.ndim != 1 or x.shape != rows[0].shape or not np.isfinite(x).all()
        for x in rows
    ):
        raise ValueError("Aligned finite source readers required")
    return np.mean(rows, axis=0), members


def endpoint(model, state, kwargs, token_ids, contrast, block=22, post=False):
    """Evaluate the original full-sequence fixed logit contrast, optionally post-attention."""
    if model.config.model_type != "qwen2" or not 0 <= block < len(model.model.layers):
        raise ValueError("Declared Qwen2 suffix required")
    x = state
    start = block
    if post:
        layer = model.model.layers[block]
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        start += 1
    for layer in model.model.layers[start:]:
        x = layer(x, **kwargs)
        x = x[0] if isinstance(x, tuple) else x
    logits = model.lm_head(model.model.norm(x[:, -1:]))[0, -1]
    return (logits[token_ids] * contrast).sum()


def endpoint_gradients(model, state, kwargs, token_ids, contrast, block=22):
    """Capture pre-attention and post-attention gradients in one actual reverse pass."""
    import torch

    captured = {}

    def hook(_module, args):
        captured["post"] = args[0]

    handle = model.model.layers[
        block
    ].post_attention_layernorm.register_forward_pre_hook(hook)
    try:
        with torch.enable_grad():
            x = state.detach().requires_grad_(True)
            score = endpoint(model, x, kwargs, token_ids, contrast, block)
            gx, gr = torch.autograd.grad(score, (x, captured["post"]))
        return (
            score.detach(),
            gx.detach(),
            gr[:, -1].detach(),
            captured["post"].detach(),
        )
    finally:
        handle.remove()


def account(actual, transported, compiled, tau):
    """Preserve the exact selection/readout error and native-dependent assemblies."""
    y, p, c, t = np.broadcast_arrays(actual, transported, compiled, tau)
    return {
        "actual": y,
        "transported": p,
        "compiled": c,
        "selection_error": p - y,
        "reader_error": c - p,
        "total_error": c - y,
        "tau": t,
        "omitted": t - y,
        "physical_patch": t + p - y,
        "compiled_swap": t + c - y,
    }
