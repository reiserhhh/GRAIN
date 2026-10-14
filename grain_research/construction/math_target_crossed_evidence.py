#!/usr/bin/env python3
"""Cross source criteria with recipient criteria on event-bound numeric evidence."""

import hashlib

import numpy as np

from grain_research.construction import math_full_attention_rows as F

BACKEND = "grain_target_crossed_hm232"


def suffix_support(rows, count):
    """Bind five identical pre-final token suffixes, never substitute event IDs."""
    if len(rows) != 5 or count < 1 or any(len(r) <= count for r in rows):
        raise ValueError("Five complete criteria and a positive suffix required")
    addresses = [list(range(len(r) - count - 1, len(r) - 1)) for r in rows]
    if (
        len({tuple(r[a] for a in ix) for r, ix in zip(rows, addresses, strict=True)})
        != 1
    ):
        raise ValueError("Carrier token identity differs between criteria")
    return addresses


def common_and_specific(values):
    """Retain criterion-average and departures in their original carrier units."""
    a = np.asarray(values, float)
    if a.ndim != 5 or a.shape[1] != 5 or not np.isfinite(a).all():
        raise ValueError("Finite layer x5criteria xKV-head xsuffix xdimension")
    common = a.mean(1, keepdims=True)
    return common, a - common


def crossed_accounts(readings):
    """Keep pair-odd/even, shared-source action, and the complete 5x5 interaction."""
    a = np.asarray(readings, float)
    if a.ndim != 4 or a.shape[1:] != (2, 6, 5) or not np.isfinite(a).all():
        raise ValueError("Finite edge x2orders x(5donors+common) x5readers")
    odd = (a[:, 0] - a[:, 1]) / 2
    even = (a[:, 0] + a[:, 1]) / 2
    # Matrix rows are recipient criteria; columns are source criteria.
    matrix = odd[:, :5].transpose(0, 2, 1)
    common_action = odd[:, 5]
    departure = matrix - common_action[:, :, None]
    grand = matrix.mean((1, 2), keepdims=True)
    reader = matrix.mean(2, keepdims=True) - grand
    source = matrix.mean(1, keepdims=True) - grand
    interaction = matrix - grand - reader - source
    return {
        "odd": odd,
        "even": even,
        "matrix": matrix,
        "common_action": common_action,
        "departure": departure,
        "grand": grand,
        "reader": reader,
        "source": source,
        "interaction": interaction,
    }


def install(model, unit, count, *, closed=False, donor=None):
    """Install a scoped full-attention observer; only final-row V additions import data.

    Source closure blocks the whole post-body query from attending to the body.
    Prefix, masks, tokens and all other edges remain. Cold imports add a numeric
    V increment to a fixed suffix, recomputing all final Q/K and probabilities.
    """
    import torch
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import eager_mask

    rows = unit["input_ids"]
    suffix = suffix_support(rows, count)
    if closed and donor is not None:
        raise ValueError("Source closure and recipient import are separate operations")
    body = unit.get("body", [])
    if closed and (not body or min(suffix[0]) <= max(body)):
        raise ValueError("Source closure needs a pre-query body")
    if model.config.model_type != "qwen2" or model.config.use_sliding_window:
        raise ValueError("Frozen full-attention Qwen2 required")
    if body and len({tuple(r[: max(body) + 1]) for r in rows}) != 1:
        raise ValueError("All criteria must have the same causal evidence prefix")
    model.config.grain_query_rows = 256
    banks, receipts = [], []

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Compute full denominators with bounded query-row memory."""
        if attention_mask is None:
            raise ValueError("Full additive causal mask required")
        changed = attention_mask
        if closed:
            changed = attention_mask.clone()
            bi = torch.tensor(body, device=key.device)
            for q, row in enumerate(rows):
                dest = torch.arange(max(body) + 1, len(row), device=key.device)
                changed[q, :, dest[:, None], bi[None, :]] = -torch.inf
        vv = torch.stack([value[q][:, ix] for q, ix in enumerate(suffix)])
        banks.append(vv.detach().float().cpu().numpy())
        digest = hashlib.sha256()
        for q, row in enumerate(rows):
            for x in (key, value):
                digest.update(
                    x[q, :, : len(row) - 1].detach().float().cpu().numpy().tobytes()
                )
        output, weights = F.full_attention_rows(
            module, query, key, value, changed, scaling=scaling, **kwargs
        )
        leakage = 0.0
        if closed:
            leakage = max(
                float(changed[q, :, max(body) + 1 : len(row), body].exp().max().item())
                for q, row in enumerate(rows)
            )
            if leakage != 0:
                raise ValueError("Closed body remains reachable")
        if donor is not None:
            from transformers.models.qwen2.modeling_qwen2 import repeat_kv

            supplied = torch.as_tensor(
                donor[module.layer_idx], dtype=value.dtype, device=value.device
            )
            if supplied.shape != vv.shape:
                raise ValueError("Numeric donor has changed carrier coordinates")
            expanded = repeat_kv(supplied, module.num_key_value_groups)
            expanded_keys = repeat_kv(key, module.num_key_value_groups)
            last = torch.tensor([len(r) - 1 for r in rows], device=query.device)
            qi = torch.arange(5, device=query.device)
            scores = (
                torch.matmul(query[qi, :, last, None], expanded_keys.transpose(2, 3))
                * scaling
            )
            scores += changed[qi, :, last, None]
            p = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)[
                :, :, 0
            ]
            before = output.clone()
            addition = torch.stack(
                [
                    (p[q][:, ix, None] * expanded[q]).sum(1)
                    for q, ix in enumerate(suffix)
                ]
            )
            output[qi, last] += addition
            before[qi, last] = output[qi, last]
            if not torch.equal(before, output):
                raise ValueError("Import altered a pre-final output")
        receipts.append(
            {
                "layer": module.layer_idx,
                "prefix_KV_hash": digest.hexdigest(),
                "closed_body_access": leakage,
                "final_only": True,
            }
        )
        return output, weights

    AttentionInterface.register(BACKEND, interface)
    AttentionMaskInterface.register(BACKEND, eager_mask)
    model.set_attn_implementation(BACKEND)
    return banks, receipts


def observe(observer, unit, count, *, closed=False, donor=None):
    """Return newly computed logits/states/banks and intervention receipts."""
    import torch

    rows = unit["input_ids"]
    device = next(observer.model.parameters()).device
    width = max(map(len, rows))
    pad = observer.tokenizer.pad_token_id
    if pad is None:
        pad = observer.tokenizer.eos_token_id
    ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows], device=device)
    mask = torch.tensor(
        [[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device
    )
    old = observer.model.config._attn_implementation
    banks, receipts = install(observer.model, unit, count, closed=closed, donor=donor)
    try:
        with torch.inference_mode():
            result = observer.model.model(
                input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
            )
            last = torch.tensor([len(r) - 1 for r in rows], device=device)
            state = result.last_hidden_state[torch.arange(5, device=device), last]
            logits = observer.model.lm_head(state).float().cpu().numpy().astype(float)
            states = state.float().cpu().numpy()
    finally:
        observer.model.set_attn_implementation(old)
    if [r["layer"] for r in receipts] != list(range(len(observer.model.model.layers))):
        raise ValueError("Exactly one observation per layer required")
    return logits, states, np.stack(banks), receipts
