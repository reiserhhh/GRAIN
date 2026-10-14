#!/usr/bin/env python3
"""Criterion-indexed two-route observations with no fitted accuracy objective."""

import hashlib

import numpy as np

ROUTES = {"closed": (0, 0), "direct": (0, 1), "relay": (1, 0), "native": (1, 1)}


def support(unit):
    """Require identical causal evidence prefixes and complete relay addresses."""
    rows, body = unit["input_ids"], unit["body"]
    if len(rows) != 6 or any(not r for r in rows):
        raise ValueError("Six nonempty contract rows required")
    if not body or body != list(range(min(body), max(body) + 1)) or min(body) < 1:
        raise ValueError("Contiguous body and prebody reference required")
    if len({tuple(r[: max(body) + 1]) for r in rows}) != 1:
        raise ValueError("Identical evidence prefixes required")
    relay = [list(range(max(body) + 1, len(r) - 1)) for r in rows]
    if any(not r for r in relay):
        raise ValueError("Nonempty pre-final relay required")
    return relay


def cut_mask(mask, unit, relay_open, direct_open):
    """Close declared source-to-destination edges, retaining all other entries."""
    import torch

    relay = support(unit)
    if relay_open not in (0, 1) or direct_open not in (0, 1):
        raise ValueError("Binary route availability required")
    if mask is None or mask.ndim != 4 or mask.shape[0] != 6:
        raise ValueError("Full six-row additive mask required")
    if mask.shape[-2] != mask.shape[-1]:
        raise ValueError("Square full-prefix attention required")
    if relay_open and direct_open:
        return mask
    changed = mask.clone()
    bi = torch.tensor(unit["body"], device=mask.device)
    for q, rows in enumerate(relay):
        destinations = ([] if relay_open else rows) + (
            [] if direct_open else [len(unit["input_ids"][q]) - 1]
        )
        ri = torch.tensor(destinations, device=mask.device, dtype=torch.long)
        changed[q, :, ri[:, None], bi[None, :]] = -torch.inf
    return changed


def route_accounts(values):
    """Keep closed, direct, relay and joint increments on each author/contract."""
    a = np.asarray(values, float)
    if a.ndim < 3 or a.shape[1:3] != (2, 2) or not np.isfinite(a).all():
        raise ValueError("Finite author xrelay xdirect xoptional-features required")
    closed, direct, relay, joint = a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1]
    return {
        "closed": closed,
        "direct_increment": direct - closed,
        "relay_increment": relay - closed,
        "joint_increment": joint - closed,
        "interaction": joint - relay - direct + closed,
        "relay_with_direct": joint - direct,
        "direct_with_relay": joint - relay,
    }


def native_batch(observer, unit, *, relay_open, direct_open):
    """Observe all six contracts under one all-layer route graph intervention."""
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    relay = support(unit)
    if observer.model.config._attn_implementation != "eager":
        raise ValueError("Inspected eager implementation required")
    rows, body = unit["input_ids"], unit["body"]
    width, device = max(map(len, rows)), next(observer.model.parameters()).device
    pad = observer.tokenizer.pad_token_id
    if pad is None:
        pad = observer.tokenizer.eos_token_id
    ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows], device=device)
    mask = torch.tensor(
        [[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device
    )
    qi = torch.arange(6, device=device)
    last = torch.tensor([len(r) - 1 for r in rows], device=device)
    bi = torch.tensor(body, device=device)
    original, receipts, calls = qwen.eager_attention_forward, [], []

    def tensor_hash(tensor):
        return hashlib.sha256(
            tensor.detach().float().cpu().numpy().tobytes()
        ).hexdigest()

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Perform actual graph surgery, not a post-hoc attention renormalization."""
        changed = cut_mask(attention_mask, unit, relay_open, direct_open)
        output, weights = original(
            module, query, key, value, changed, scaling=scaling, **kwargs
        )
        shared = max(
            float((x[:, :, bi] - x[:1, :, bi]).abs().max().item()) for x in (key, value)
        )
        rmax = max(
            float(weights[q, :, r][:, :, bi].abs().max().item())
            for q, r in enumerate(relay)
        )
        dmax = float(weights[qi, :, last][:, :, bi].abs().max().item())
        if (
            shared > 3e-4
            or (not relay_open and rmax != 0)
            or (not direct_open and dmax != 0)
        ):
            raise ValueError("Prefix or route graph intervention failed")
        digest = hashlib.sha256()
        for q, r in enumerate(relay):
            for x in (key, value):
                digest.update(x[q, :, r].detach().float().cpu().numpy().tobytes())
        receipts.append(
            {
                "layer": module.layer_idx,
                "shared_body_max": shared,
                "relay_body_weight_max": rmax,
                "final_body_weight_max": dmax,
                "body_key_hash": tensor_hash(key[0, :, bi]),
                "body_value_hash": tensor_hash(value[0, :, bi]),
                "relay_KV_hash": digest.hexdigest(),
            }
        )
        calls.append(module.layer_idx)
        return output, weights

    try:
        qwen.eager_attention_forward = interface
        with torch.inference_mode():
            result = observer.model.model(
                input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
            )
            states = result.last_hidden_state[qi, last]
            logits = observer.model.lm_head(states).float().cpu().numpy().astype(float)
            states = states.detach().float().cpu().numpy()
    finally:
        qwen.eager_attention_forward = original
    if calls != list(range(len(observer.model.model.layers))):
        raise ValueError("Each frozen layer must act exactly once")
    return logits, states, receipts
