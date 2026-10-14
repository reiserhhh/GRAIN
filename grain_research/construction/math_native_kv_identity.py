#!/usr/bin/env python3
"""Independent native-history K/V ownership and finite diagonal restriction."""

from __future__ import annotations

import itertools

import numpy as np


def characters(cube):
    """Return all16 characters of a complete b/k/v/p observation cube."""
    x = np.asarray(cube, dtype=np.float64)
    if x.shape[:4] != (2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete finite b/k/v/p support required")
    signs = np.array(list(itertools.product((-1, 1), repeat=4)))
    flat = x.reshape(16, *x.shape[4:])
    return np.stack(
        [
            np.einsum(
                "a,a...->...",
                np.prod(signs[:, [j for j in range(4) if mask & (1 << j)]], axis=1),
                flat,
            )
            / 16
            for mask in range(16)
        ]
    )


def diagonal(cube):
    """Restrict natural histories to k=v=o without discarding off-diagonal data."""
    x = np.asarray(cube)
    if x.shape[:4] != (2, 2, 2, 2):
        raise ValueError("A four-role observation cube is required")
    return np.stack([x[:, o, o] for o in range(2)], axis=1)


def restrict_coefficients(gamma):
    """Sum characters that coalesce under the diagonal subgroup k=v=o."""
    x = np.asarray(gamma, dtype=np.float64)
    if x.shape[0] != 16 or not np.isfinite(x).all():
        raise ValueError("All16 character coefficients required")
    beta = np.zeros((8, *x.shape[1:]))
    for mask in range(16):
        target = (mask & 1) | (((bool(mask & 2)) ^ (bool(mask & 4))) << 1)
        target |= (bool(mask & 8)) << 2
        beta[target] += x[mask]
    return beta


def suffix_interchange(
    original,
    query,
    key,
    value,
    mask,
    scaling,
    body,
    history,
    donor_k,
    donor_v,
    module,
    **kwargs,
):
    """Compute donor attention on suffix rows only; preserve original body rows."""
    import torch

    native, native_weights = original(
        module, query, key, value, mask, scaling=scaling, **kwargs
    )
    k, v = key.clone(), value.clone()
    expected_k = torch.as_tensor(donor_k, device=k.device, dtype=k.dtype)[None]
    expected_v = torch.as_tensor(donor_v, device=v.device, dtype=v.dtype)[None]
    k[:, :, history] = expected_k
    v[:, :, history] = expected_v
    if not torch.equal(k[:, :, history], expected_k.expand(key.shape[0], -1, -1, -1)):
        raise ValueError("Actual historical K donor was not installed")
    if not torch.equal(v[:, :, history], expected_v.expand(value.shape[0], -1, -1, -1)):
        raise ValueError("Actual historical V donor was not installed")
    changed, changed_weights = original(
        module, query, k, v, mask, scaling=scaling, **kwargs
    )
    result, weights = native.clone(), native_weights.clone()
    result[:, body:] = changed[:, body:]
    weights[:, :, body:] = changed_weights[:, :, body:]
    if not torch.equal(result[:, :body], native[:, :body]):
        raise ValueError("Interchange changed prefix attention rows")
    return result, weights


def native_batch(observer, row, queries, *, donors=None):
    """Capture or interchange rotated historical K/V at all native decoder blocks."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    from grain_research.construction import math_native_access as A

    original = qwen.eager_attention_forward
    depth = len(observer.model.model.layers)
    if observer.model.config._attn_implementation != "eager":
        raise ValueError("Scoped operation requires the inspected eager interface")
    capture, calls, spread = {}, [], 0.0
    body, history = len(row["input_ids"]), row["history"]

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Keep stock body computations and record actual post-RoPE carriers."""
        nonlocal spread
        i = module.layer_idx
        calls.append(i)
        for name, tensor in (("key", key), ("value", value)):
            h = tensor[:, :, history].float().cpu().numpy()
            spread = max(spread, float(np.max(abs(h - h[:1]))))
            if donors is None:
                capture[name, i] = h[0].copy()
        if donors is None:
            return original(
                module, query, key, value, attention_mask, scaling=scaling, **kwargs
            )
        return suffix_interchange(
            original,
            query,
            key,
            value,
            attention_mask,
            scaling,
            body,
            history,
            donors[0][i],
            donors[1][i],
            module,
            **kwargs,
        )

    try:
        qwen.eager_attention_forward = interface
        reads, arrays = A.native_batch(
            observer, row, queries["suffixes"], queries["alphabet"], route=1
        )
    finally:
        qwen.eager_attention_forward = original
    if calls != list(range(depth)):
        raise ValueError("All inspected blocks must execute the historical operation")
    arrays["cache_spread"] = np.asarray(spread)
    arrays["interchange_blocks"] = np.asarray(calls)
    if donors is None:
        arrays["history_key"] = np.stack([capture["key", i] for i in range(depth)])
        arrays["history_value"] = np.stack([capture["value", i] for i in range(depth)])
    arrays["logits"] = np.array([r["candidate_logits"] for r in reads])
    arrays["normalizer"] = np.array([r["full_vocab_log_normalizer"] for r in reads])
    arrays["readings"] = A.units(arrays["logits"], arrays["normalizer"])
    return arrays
