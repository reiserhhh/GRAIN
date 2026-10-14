#!/usr/bin/env python3
"""Forward evidence-to-query edge cuts, separate from derivative attribution."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_target_selector_exchange as S


def relay_support(unit):
    """Return every post-body, pre-final address; reject hidden support gaps."""
    rows, body = unit["input_ids"], unit["body"]
    if len(rows) != 5 or any(not r for r in rows):
        raise ValueError("Five nonempty physical query rows required")
    if not body or body != list(range(min(body), max(body) + 1)):
        raise ValueError("Contiguous complete body support required")
    relay = [list(range(max(body) + 1, len(r) - 1)) for r in rows]
    if min(body) < 0 or any(not r for r in relay):
        raise ValueError("Nonempty pre-answer relay required for every query")
    prefix = [r[: max(body) + 1] for r in rows]
    if any(r != prefix[0] for r in prefix):
        raise ValueError("Identical causal body prefixes required")
    return relay


def cut_mask(attention_mask, body, relay):
    """Clone the native additive mask and forbid body-to-relay attention edges."""
    import torch

    if (
        attention_mask is None
        or attention_mask.ndim != 4
        or attention_mask.shape[0] != len(relay)
        or attention_mask.shape[-2] != attention_mask.shape[-1]
    ):
        raise ValueError("Full native batched additive causal mask required")
    changed = attention_mask.clone()
    bi = torch.tensor(body, device=changed.device)
    for q, rows in enumerate(relay):
        ri = torch.tensor(rows, device=changed.device)
        changed[q, :, ri[:, None], bi[None, :]] = -torch.inf
    return changed


def factorial_actions(corners):
    """Keep all four observations and signed relay/selection/interaction effects."""
    a = np.asarray(corners, float)
    if a.ndim < 4 or a.shape[-4:-2] != (2, 2) or not np.isfinite(a).all():
        raise ValueError("Finite (...,relay,selector,target,unit) corners required")
    d00, d01, d10, d11 = (
        a[..., i, j, :, :] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    relay, selection = d10 - d00, d01 - d00
    interaction = d11 - d10 - d01 + d00
    return {
        "intact_native": d00,
        "intact_foreign": d01,
        "cut_native": d10,
        "cut_foreign": d11,
        "relay_delta": relay,
        "selection_delta": selection,
        "interaction": interaction,
        "joint_delta": d11 - d00,
    }


def native_batch(observer, unit, mapping=None, donor=None, *, cut=True, fixed_c=False):
    """Run all-layer relay cuts with optional frozen final-slot selector exchange.

    The final row retains direct body access. In foreign mode C is explicitly
    left native under the cut, rather than receiving an intact C distribution.
    No historical donor array is modified. Nested eager hooks are restored even
    on failure; this serial interface must not run concurrently with other hooks.
    """
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    relay, body = relay_support(unit), unit["body"]
    if fixed_c and (mapping is None or S.permutation(mapping, 5)[3] != 3):
        raise ValueError("Fixed C requires its own donor assignment")
    donor_copy = None if donor is None else np.array(donor, copy=True)
    if fixed_c and donor_copy is None:
        raise ValueError("Fixed C is only an exchange control")
    original, diagnostics = qwen.eager_attention_forward, []

    def relay_interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Alter only declared edges before the sealed selector interface acts."""
        changed = cut_mask(attention_mask, body, relay) if cut else attention_mask
        output, weights = original(
            module, query, key, value, changed, scaling=scaling, **kwargs
        )
        bi = torch.tensor(body, device=weights.device)
        leak = max(
            float(weights[q, :, rows][:, :, bi].abs().max().item())
            for q, rows in enumerate(relay)
        )
        if cut and leak != 0:
            raise ValueError("Body-to-relay edge cut was not installed")
        if fixed_c:
            last = len(unit["input_ids"][3]) - 1
            p = weights[3, :, last][:, bi]
            expected = (weights.shape[1], len(body))
            if p.shape != expected:
                raise ValueError("Inspected C selector support changed")
            s = p / p.sum(-1, keepdim=True)
            donor_copy[module.layer_idx, 3] = s.detach().float().cpu().numpy()
        diagnostics.append(
            {
                "layer": module.layer_idx,
                "relay_body_weight_max": leak,
                "cut": bool(cut),
                "relay_rows": [len(r) for r in relay],
            }
        )
        return output, weights

    try:
        qwen.eager_attention_forward = relay_interface
        result = S.native_batch(observer, unit, mapping=mapping, donor=donor_copy)
    finally:
        qwen.eager_attention_forward = original
    if len(diagnostics) != len(observer.model.model.layers):
        raise ValueError("Complete all-layer relay intervention required")
    return (*result, diagnostics)
