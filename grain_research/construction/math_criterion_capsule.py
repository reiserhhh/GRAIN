#!/usr/bin/env python3
"""Source-observed evidence increments transported to an excerpt-free reader."""

import hashlib

import numpy as np

from grain_research.construction import math_contract_routes as G

MODES = ("aligned", "swapped", "common")


def suffix_support(rows, count):
    """Locate identical pre-answer tokens, without treating addresses as events."""
    if len(rows) != 6 or count < 1 or any(len(r) <= count for r in rows):
        raise ValueError("Six complete contracts and positive suffix required")
    addresses = [list(range(len(r) - 1 - count, len(r) - 1)) for r in rows]
    if len({tuple(r[i] for i in a) for r, a in zip(rows, addresses, strict=True)}) != 1:
        raise ValueError("Suffix token identity is required for transport")
    return addresses


def capsule(native, closed):
    """Return paired S-minus-T body-access V increments; retain both baselines."""
    n, b = np.asarray(native, float), np.asarray(closed, float)
    if n.shape != b.shape or n.ndim != 6 or n.shape[0] != 2 or n.shape[2] != 6:
        raise ValueError("Two arms xlayer x6contracts xKV-head xsuffix xdimension")
    if not np.isfinite(n).all() or not np.isfinite(b).all():
        raise ValueError("Finite observed banks required")
    return (n[0] - b[0]) - (n[1] - b[1])


def condition_values(values, mode):
    """Keep aligned, exchange criteria within each name, or retain their common part."""
    v = np.asarray(values)
    if v.ndim != 5 or v.shape[1] != 6 or not np.isfinite(v).all():
        raise ValueError("Finite layer x6contracts xKV-head xsuffix xdimension")
    if mode == "aligned":
        return v.copy()
    if mode == "swapped":
        return v[:, [1, 0, 3, 2, 5, 4]].copy()
    if mode == "common":
        shaped = v.reshape(v.shape[0], 3, 2, *v.shape[2:])
        return (
            np.broadcast_to(shaped.mean(2, keepdims=True), shaped.shape)
            .reshape(v.shape)
            .copy()
        )
    raise ValueError("Declared criterion mapping required")


def native_batch(observer, unit, count, *, closed=False, values=None):
    """Capture V or inject one capsule into a fresh naturally routed reader.

    Capture keeps full native/body-blind prefix computation. Injection changes
    only final-row attention output through suffix values, never prefix states,
    K, masks, or cached source probabilities. Final probabilities are recomputed
    on the recipient at every layer, including feedback from previous layers.
    """
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    rows = unit["input_ids"]
    suffix = suffix_support(rows, count)
    if closed and values is not None:
        raise ValueError("Capture and recipient injection are distinct operations")
    if closed and any(min(a) <= max(unit["body"]) for a in suffix):
        raise ValueError("Carrier must be after source evidence")
    if observer.model.config._attn_implementation != "eager":
        raise ValueError("Eager frozen observer required")
    device = next(observer.model.parameters()).device
    width = max(map(len, rows))
    pad = observer.tokenizer.pad_token_id
    if pad is None:
        pad = observer.tokenizer.eos_token_id
    ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows], device=device)
    mask = torch.tensor(
        [[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device
    )
    ii = torch.arange(6, device=device)
    last = torch.tensor([len(r) - 1 for r in rows], device=device)
    original, banks, receipts = qwen.eager_attention_forward, [], []

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Use natural recipient probabilities, not an inherited final reading law."""
        changed = G.cut_mask(attention_mask, unit, 0, 0) if closed else attention_mask
        output, weights = original(
            module, query, key, value, changed, scaling=scaling, **kwargs
        )
        vv = torch.stack([value[q][:, a] for q, a in enumerate(suffix)])
        banks.append(vv.detach().float().cpu().numpy())
        prefix = hashlib.sha256()
        for q, length in enumerate(map(len, rows)):
            for x in (key, value):
                prefix.update(
                    x[q, :, : length - 1].detach().float().cpu().numpy().tobytes()
                )
        body_digest = hashlib.sha256()
        if "body" in unit:
            bi = torch.tensor(unit["body"], device=device)
            shared = max(
                float((x[:, :, bi] - x[:1, :, bi]).abs().max().item())
                for x in (key, value)
            )
            if shared > 3e-4:
                raise ValueError("Contract changed the causal body prefix")
            for x in (key, value):
                body_digest.update(x[0, :, bi].detach().float().cpu().numpy().tobytes())
        before = output.clone() if values is not None else None
        if values is not None:
            donor = torch.as_tensor(
                values[module.layer_idx], dtype=value.dtype, device=device
            )
            if donor.shape != vv.shape:
                raise ValueError("Frozen capsule and recipient carrier differ")
            expanded = qwen.repeat_kv(donor, module.num_key_value_groups)
            pp = weights[ii, :, last]
            addition = torch.stack(
                [(pp[q][:, a, None] * expanded[q]).sum(1) for q, a in enumerate(suffix)]
            )
            output[ii, last] += addition
            before[ii, last] = output[ii, last]
            if float((before - output).abs().max().item()) != 0:
                raise ValueError("Injection changed a pre-final output")
        leak = 0.0
        if closed:
            bi = torch.tensor(unit["body"], device=device)
            relay = G.support(unit)
            leak = max(
                float(
                    weights[q, :, a + [len(rows[q]) - 1]][:, :, bi].abs().max().item()
                )
                for q, a in enumerate(relay)
            )
            if leak != 0:
                raise ValueError("Source closed graph leaked")
        receipts.append(
            {
                "layer": module.layer_idx,
                "prefix_KV_hash": prefix.hexdigest(),
                "source_body_KV_hash": body_digest.hexdigest()
                if "body" in unit
                else None,
                "closed_body_weight_max": leak,
                "final_p_hash": hashlib.sha256(
                    weights[ii, :, last].detach().float().cpu().numpy().tobytes()
                ).hexdigest(),
            }
        )
        return output, weights

    try:
        qwen.eager_attention_forward = interface
        with torch.inference_mode():
            result = observer.model.model(
                input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
            )
            states = result.last_hidden_state[ii, last]
            logits = observer.model.lm_head(states).float().cpu().numpy().astype(float)
            states = states.detach().float().cpu().numpy()
    finally:
        qwen.eager_attention_forward = original
    if [r["layer"] for r in receipts] != list(range(len(observer.model.model.layers))):
        raise ValueError("Exactly one call per layer required")
    return logits, states, np.stack(banks), receipts
