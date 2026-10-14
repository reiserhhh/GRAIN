#!/usr/bin/env python3
"""Fixed evidence under explicit diagnostic laws, not fitted personality scores."""

from __future__ import annotations

import numpy as np


def posterior_from_counts(counts, option):
    """Invert a stipulated two-category population table for one actual choice."""
    x = np.asarray(counts, dtype=float)
    if (
        x.shape != (2, 2)
        or not np.isfinite(x).all()
        or (x <= 0).any()
        or (x != np.floor(x)).any()
        or option not in (0, 1)
    ):
        raise ValueError(
            "Positive integer H/L by R/S counts and a valid option required"
        )
    return float(x[0, option] / x[:, option].sum())


def law_cube(positive, negative):
    """Return all b/c posterior corners, with b=-1 meaning option S."""
    return np.array(
        [
            [posterior_from_counts(counts, option) for counts in (negative, positive)]
            for option in (1, 0)
        ]
    )


def characters(corners):
    """Preserve the constant, behavior, reference and behavior/reference terms."""
    x = np.asarray(corners, dtype=float)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete finite b/c support required")
    s = np.array([-1.0, 1.0])
    return np.stack(
        (
            x.mean(axis=(0, 1)),
            np.einsum("b,bc...->...", s, x) / 4,
            np.einsum("c,bc...->...", s, x) / 4,
            np.einsum("b,c,bc...->...", s, s, x) / 4,
        )
    )


def author_contrasts(corners, eta=0.5):
    """Compare declared positive choice distributions, separately by reference."""
    x = np.asarray(corners, dtype=float)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all() or not 0 < eta < 1:
        raise ValueError("Two behaviors/references and an interior mixture required")
    return eta * (x[1] - x[0])


def intervention_parts(fields):
    """Keep K, V and their interaction for recipient/K/V/KV endpoint arrays."""
    x = np.asarray(fields, dtype=float)
    if x.shape[0] != 4 or not np.isfinite(x).all():
        raise ValueError("Complete finite recipient/K/V/KV fields required")
    return np.stack((x[1] - x[0], x[2] - x[0], x[3] - x[1] - x[2] + x[0]))


def install_reference(base_k, base_v, donor_k, donor_v, indices, mode):
    """Install only registered reference rows; leave every other row untouched."""
    from grain_research.construction.math_causal_valuation import install_cause

    return install_cause(base_k, base_v, donor_k, donor_v, indices, mode)


def native_batch(observer, row, queries, *, donors=None):
    """Read with stock permissions and optionally swap reference K/V in suffixes.

    Body relay states remain the recipient's, so swapping the reference does not
    assert equivalence to regenerating a complete opposite-reference body.
    """
    import transformers.models.qwen2.modeling_qwen2 as qwen

    from grain_research.construction import math_native_access as A
    from grain_research.construction.math_native_kv_identity import suffix_interchange

    original = qwen.eager_attention_forward
    depth = len(observer.model.model.layers)
    if observer.model.config._attn_implementation != "eager":
        raise ValueError("The frozen eager native interface is required")
    capture, calls, spread = {}, [], 0.0
    body, history = len(row["input_ids"]), row["history"]

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        """Capture causal historical carriers while preserving prefix attention."""
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
            observer, row, queries["suffixes"], queries["alphabet"], route=7
        )
    finally:
        qwen.eager_attention_forward = original
    if calls != list(range(depth)):
        raise ValueError("All frozen decoder blocks must execute")
    arrays["cache_spread"] = np.asarray(spread)
    arrays["interchange_blocks"] = np.asarray(calls)
    if donors is None:
        arrays["history_key"] = np.stack([capture["key", i] for i in range(depth)])
        arrays["history_value"] = np.stack([capture["value", i] for i in range(depth)])
    arrays["logits"] = np.array([r["candidate_logits"] for r in reads])
    arrays["normalizer"] = np.array([r["full_vocab_log_normalizer"] for r in reads])
    arrays["readings"] = A.units(arrays["logits"], arrays["normalizer"])
    return arrays
