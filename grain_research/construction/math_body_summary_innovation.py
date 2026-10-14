#!/usr/bin/env python3
"""Template-preserving contrasts of observed body-access summary values."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_query_relay_cut as Q
from grain_research.construction import math_relay_value_summary as V
from grain_research.construction import math_target_selector_exchange as S


def increment_donor(native, blind, mode, mapping=None):
    """Retain the recipient baseline while changing only the body-access increment."""
    n, b = np.asarray(native, float), np.asarray(blind, float)
    if n.shape != b.shape or n.ndim != 5 or n.shape[1] != 5:
        raise ValueError("Matching layer/query/KV-head/suffix/value banks required")
    if not np.isfinite(n).all() or not np.isfinite(b).all():
        raise ValueError("Finite native and blind banks required")
    increment = n - b
    if mode == "self":
        return b + increment
    if mode == "zero":
        return b.copy()
    if mode == "common":
        return b + increment.mean(axis=1, keepdims=True)
    if mode == "foreign":
        return b + increment[:, S.permutation(mapping, 5)]
    raise ValueError("Declared zero/common/foreign/self increment required")


def blind_batch(observer, unit):
    """Capture B on the native support with every body-to-prefinal edge cut.

    Nested serial eager hooks delegate to the sealed native V observer. Every
    hook is restored on success and failure. Final body access is not cut.
    """
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    relay = Q.relay_support(unit)
    body = unit["body"]
    original, receipts = qwen.eager_attention_forward, []

    def cut(module, query, key, value, attention_mask, scaling, **kwargs):
        """Mask all body-to-relay edges before capturing this layer's suffix V."""
        output, weights = original(
            module,
            query,
            key,
            value,
            Q.cut_mask(attention_mask, body, relay),
            scaling=scaling,
            **kwargs,
        )
        bi = torch.tensor(body, device=weights.device)
        leak = max(
            float(weights[q, :, ri][:, :, bi].abs().max().item())
            for q, ri in enumerate(relay)
        )
        if leak != 0:
            raise ValueError("Body-to-relay blind capture failed")
        receipts.append({"layer": module.layer_idx, "relay_body_weight_max": leak})
        return output, weights

    try:
        qwen.eager_attention_forward = cut
        result = V.native_batch(observer, unit)
    finally:
        qwen.eager_attention_forward = original
    if [r["layer"] for r in receipts] != list(range(len(observer.model.model.layers))):
        raise ValueError("One complete blind capture per layer required")
    return (*result, receipts)


def reader_batch(observer, unit, native_bank, blind_values, mode, mapping=None):
    """Apply the observed increment to the original prefix under frozen native p."""
    donor = increment_donor(native_bank["values"], blind_values, mode, mapping)
    # Override donor construction only; retain the native bank for prefix guards.
    original = V.summary_values

    def supplied(values, selected, supplied_mapping=None):
        """Supply one immutable donor without replacing the native prefix guard."""
        if selected != "self" or supplied_mapping is not None:
            raise ValueError("Explicit donor replay only")
        if np.asarray(values).shape != donor.shape:
            raise ValueError("Native and donor support differ")
        return donor.copy()

    try:
        V.summary_values = supplied
        return V.native_batch(observer, unit, native_bank, "self")
    finally:
        V.summary_values = original
