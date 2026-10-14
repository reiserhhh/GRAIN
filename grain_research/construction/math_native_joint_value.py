#!/usr/bin/env python3
"""A finite bridge from joint value coding to native decoder decisions."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import logsumexp

from grain_research.construction.math_joint_reading import BASIS, coefficients


def joint_edits(values, key):
    """Remove only the joint character; return a deterministic head-norm control."""
    v = np.asarray(values, float)
    if v.ndim != 4 or v.shape[0] != 4 or min(v.shape) == 0:
        raise ValueError("Four aligned token/GQA-head/value corners required")
    joint = coefficients(v)[3]
    seed = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "little")
    random = np.random.default_rng(seed).normal(size=joint.shape)
    norm = np.linalg.norm(joint, axis=(0, 2), keepdims=True)
    random *= norm / np.linalg.norm(random, axis=(0, 2), keepdims=True)
    sign = BASIS[3, :, None, None, None]
    return -sign * joint, -sign * random


def edit_values(output, positions, delta):
    """Edit selected physical value rows; leave all unselected rows identical."""
    import torch

    p = list(positions)
    if (
        output.ndim != 3
        or output.shape[0] != 1
        or not p
        or len(set(p)) != len(p)
        or any(not isinstance(i, int) or not 0 <= i < output.shape[1] for i in p)
    ):
        raise ValueError("Unique selected rows in a single full prompt required")
    d = torch.as_tensor(delta, dtype=output.dtype, device=output.device)
    if d.shape != (len(p), output.shape[2]) or not torch.isfinite(d).all():
        raise ValueError("Finite value edit must match selected projection rows")
    updated = output.clone()
    updated[0, p] = output[0, p] + d
    return updated


def native_units(logits, normalizer):
    """Keep conditional digit score, grouped log odds and compliance distinct."""
    z = np.asarray(logits, float)
    if z.shape != (10,) or not np.isfinite(z).all() or not np.isfinite(normalizer):
        raise ValueError("Ten finite fixed digit logits and full normalizer required")
    norm = logsumexp(z)
    p = np.exp(z - norm)
    return np.array(
        [
            p @ (np.arange(10) / 9),
            logsumexp(z[5:]) - logsumexp(z[:5]),
            np.exp(norm - normalizer),
            -(p * (z - norm)).sum(),
        ]
    )


def native_forward(observer, row, suffix, alphabet, delta, layer=17):
    """Score one full prompt with scoped pad masks and a current-value-only edit."""
    import torch

    from grain_research.vector_mechanisms.direct_traits import native_readout
    from grain_research.vector_mechanisms.prefix import rotate_half

    ids = row["input_ids"] + list(suffix)
    if len(ids) > observer.max_tokens:
        raise ValueError("Do not truncate complete native evidence")
    current = row["current"]
    attention = observer.model.model.layers[layer].self_attn
    dim = attention.head_dim
    if delta is not None and np.asarray(delta).shape != (
        len(current),
        attention.k_proj.out_features // dim,
        dim,
    ):
        raise ValueError("Current physical GQA value coordinates changed")
    handles, saved, masks = [], {}, []

    def mask_hook(index):
        def change(_module, args, kwargs):
            mask = kwargs["attention_mask"].clone()
            mask[:, :, :, row["padding"]] = torch.finfo(mask.dtype).min
            if mask.shape[-2:] != (len(ids), len(ids)):
                raise ValueError("Full explicit causal attention mask required")
            if index == 0:
                support = (mask[0, 0] == 0).cpu().numpy()
                expected = np.tri(len(ids), dtype=bool)
                expected[:, row["padding"]] = False
                if not np.array_equal(support, expected):
                    raise ValueError(
                        "Actual native permissions differ from the declared pad/causal support"
                    )
                saved["support_hash"] = np.frombuffer(
                    hashlib.sha256(support.tobytes()).digest(), dtype=np.uint8
                ).copy()
            masks.append(index)
            if index == layer:
                cos, sin = kwargs["position_embeddings"]
                saved["cos"] = cos[0, current].float().cpu().numpy().copy()
                saved["sin"] = sin[0, current].float().cpu().numpy().copy()
            return args, dict(kwargs, attention_mask=mask)

        return change

    def key_hook(_module, _args, output):
        saved["raw_key"] = output[0, current].float().cpu().numpy().copy()

    def value_hook(_module, _args, output):
        saved["before"] = output[0, current].float().cpu().numpy().copy()
        if delta is not None:
            output = edit_values(
                output, current, np.asarray(delta).reshape(len(current), -1)
            )
        saved["after"] = output[0, current].float().cpu().numpy().copy()
        return output

    try:
        for i, block in enumerate(observer.model.model.layers):
            handles.append(
                block.self_attn.register_forward_pre_hook(
                    mask_hook(i), with_kwargs=True
                )
            )
        handles.append(attention.k_proj.register_forward_hook(key_hook))
        handles.append(attention.v_proj.register_forward_hook(value_hook))
        tokens = torch.tensor([ids], device=next(observer.model.parameters()).device)
        with torch.inference_mode():
            output = observer.model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        score = native_readout(output.logits[0, -1].float().cpu().numpy(), alphabet)
    finally:
        for handle in handles:
            handle.remove()
    observer.calls += 1
    if masks != list(range(len(observer.model.model.layers))):
        raise RuntimeError("Incomplete actual pad-mask execution")
    k = saved["raw_key"].reshape(len(current), -1, dim)
    cos, sin = saved["cos"][:, None], saved["sin"][:, None]
    return score, {
        "k": k * cos + rotate_half(k) * sin,
        "before": saved["before"].reshape(k.shape),
        "after": saved["after"].reshape(k.shape),
        "masked_blocks": np.asarray(masks),
        "support_hash": saved["support_hash"],
    }
