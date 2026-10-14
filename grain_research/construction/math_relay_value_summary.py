#!/usr/bin/env python3
"""Identical-surface value summaries with an explicitly frozen reading law."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_target_selector_exchange as S


def common_suffix(unit):
    """Return maximal identical pre-answer suffix at query-relative addresses."""
    rows, body = unit["input_ids"], unit["body"]
    if len(rows) != 5 or not body or any(len(r) < 3 for r in rows):
        raise ValueError("Five complete query rows and body required")
    pre = [r[:-1] for r in rows]
    count = 0
    for offset in range(1, min(map(len, pre)) + 1):
        if len({r[-offset] for r in pre}) != 1:
            break
        count = offset
    suffix = [list(range(len(r) - 1 - count, len(r) - 1)) for r in rows]
    if count == 0 or any(min(r) <= max(body) for r in suffix):
        raise ValueError("Nonempty identical suffix must be strictly after body")
    return suffix


def summary_values(values, mode, mapping=None):
    """Keep native/common/foreign query summaries in exactly the same V units."""
    v = np.asarray(values)
    if v.ndim != 5 or v.shape[1] != 5 or not np.isfinite(v).all():
        raise ValueError("Finite layer/query/KV-head/suffix/dimension values required")
    if mode == "self":
        return v.copy()
    if mode == "common":
        return np.broadcast_to(v.mean(axis=1, keepdims=True), v.shape).copy()
    if mode == "foreign":
        return v[:, S.permutation(mapping, 5)].copy()
    raise ValueError("Declared common, foreign or identity summary required")


def fixed_value_output(probabilities, values, suffix, donor):
    """Compute the frozen final-row contraction without modifying probabilities."""
    p, v, d = (np.asarray(a, float) for a in (probabilities, values, donor))
    if p.ndim != 3 or v.shape[:3] != p.shape or v.ndim != 4:
        raise ValueError("query/head/token probabilities and values required")
    if len(suffix) != len(p) or d.shape != (*p.shape[:2], len(suffix[0]), v.shape[-1]):
        raise ValueError("Aligned query/head/suffix donor required")
    if not all(np.isfinite(a).all() for a in (p, v, d)) or (p < 0).any():
        raise ValueError("Finite positive reading support required")
    if not np.allclose(p.sum(-1), 1, atol=2e-6, rtol=0):
        raise ValueError("Normalized frozen probabilities required")
    changed = v.copy()
    for q, rows in enumerate(suffix):
        changed[q][:, rows] = d[q]
    return np.einsum("qhi,qhid->qhd", p, changed)


def native_batch(observer, unit, bank=None, mode="native", mapping=None):
    """Capture or read a value bank; change only the final attention output row.

    All final-row probabilities in replay are the cached native instrument.
    Earlier rows retain native outputs and therefore native K/V. C is a foreign
    identity control but is not exempted from the common-summary construction.
    This serial eager hook is restored on every exit, including failed forwards.
    """
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    if (bank is None) != (mode == "native"):
        raise ValueError("Native capture or explicit frozen bank replay required")
    suffix, rows = common_suffix(unit), unit["input_ids"]
    last = [len(r) - 1 for r in rows]
    original = qwen.eager_attention_forward
    saved_p, saved_v, receipts = [], [], []
    donors = None if bank is None else summary_values(bank["values"], mode, mapping)

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Contract cached probabilities with native or substituted suffix values."""
        output, weights = original(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )
        layer, device = module.layer_idx, value.device
        ii = torch.arange(5, device=device)
        li = torch.tensor(last, device=device)
        pp = weights[ii, :, li].clone()
        vv = torch.stack([value[q][:, r] for q, r in enumerate(suffix)])
        receipt = {
            "layer": layer,
            "fixed_probability_error": 0.0,
            "prefix_V_error": 0.0,
        }
        if bank is None:
            saved_p.append(pp.detach().float().cpu().numpy())
            saved_v.append(vv.detach().float().cpu().numpy())
        else:
            fixed = torch.as_tensor(bank["probabilities"][layer], device=device)
            stored = torch.as_tensor(bank["values"][layer], device=device)
            receipt["prefix_V_error"] = float((vv - stored).abs().max().item())
            if fixed.shape != pp.shape or receipt["prefix_V_error"] > 3e-4:
                raise ValueError("Frozen prefix or query reading support changed")
            expanded = qwen.repeat_kv(value, module.num_key_value_groups)
            desired = torch.as_tensor(donors[layer], device=device)
            desired = qwen.repeat_kv(desired, module.num_key_value_groups)
            # Replace the complete final-row contraction, not only one head.
            final = torch.matmul(fixed.unsqueeze(-2), expanded).squeeze(-2)
            for q, ri in enumerate(suffix):
                final[q] += (
                    fixed[q][:, ri].unsqueeze(-1) * (desired[q] - expanded[q][:, ri])
                ).sum(1)
            output[ii, li] = final
            weights[ii, :, li] = fixed
            receipt["fixed_probability_error"] = float(
                (weights[ii, :, li] - fixed).abs().max().item()
            )
        receipts.append(receipt)
        return output, weights

    try:
        qwen.eager_attention_forward = interface
        result = S.native_batch(observer, unit)
    finally:
        qwen.eager_attention_forward = original
    if [r["layer"] for r in receipts] != list(range(len(observer.model.model.layers))):
        raise ValueError("All layers must execute exactly once")
    captured = (
        None
        if bank is not None
        else {"probabilities": np.stack(saved_p), "values": np.stack(saved_v)}
    )
    return (*result, captured, receipts)
