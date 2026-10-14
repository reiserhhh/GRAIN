#!/usr/bin/env python3
"""Causal suffix-access programs and lossless finite route observations."""

from __future__ import annotations

import hashlib

import numpy as np


def partition(row):
    """Separate prior system tokens from the post-history contextual relay."""
    length = len(row["input_ids"])
    h, c = list(row["history"]), list(row["current"])
    s = list(range(min(h)))
    b = sorted(set(range(length)) - set(s + h + c))
    groups = {"S": s, "H": h, "B": b, "C": c}
    flat = s + h + b + c
    if len(flat) != length or sorted(flat) != list(range(length)):
        raise ValueError("Body partition must be disjoint and exhaustive")
    if not b or not c or max(h) >= min(b) or max(b) >= min(c):
        raise ValueError("Need a prior system, history, post-history relay, current")
    if not set(row["padding"]) <= set(h):
        raise ValueError("This fixed support only admits history-slot padding")
    return groups


def permissions(row, suffix_length, execution_length, route):
    """Return actual causal permissions, cutting every suffix-to-body edge."""
    if route not in range(8) or suffix_length < 1:
        raise ValueError("Three carrier bits and a complete suffix required")
    body = len(row["input_ids"])
    endpoint = body + suffix_length
    if endpoint > execution_length:
        raise ValueError("Truncated native suffix")
    support = np.tri(execution_length, dtype=bool)
    support[:, row["padding"]] = False
    support[:, endpoint:] = False
    groups = partition(row)
    for bit, name in enumerate(("H", "B", "C")):
        if not route & (1 << bit):
            support[body:, groups[name]] = False
    return support


def mobius(fields):
    """Compute all eight signed access components without dropping interactions."""
    values = np.asarray(fields, dtype=np.float64)
    if values.shape[0] != 8 or not np.isfinite(values).all():
        raise ValueError("All eight finite access fields required")
    result = np.empty_like(values)
    for subset in range(8):
        result[subset] = sum(
            (-1) ** (subset.bit_count() - sub.bit_count()) * values[sub]
            for sub in range(8)
            if sub & subset == sub
        )
    return result


def reconstruct(components):
    """Reconstruct every access field, not just the full-route endpoint."""
    values = np.asarray(components, dtype=np.float64)
    return np.stack(
        [sum(values[sub] for sub in range(8) if sub & mask == sub) for mask in range(8)]
    )


def characters(corners):
    """Expand polarity, acting person and queried person on complete support."""
    value = np.asarray(corners, dtype=np.float64)
    if value.shape[:3] != (2, 2, 2) or not np.isfinite(value).all():
        raise ValueError("All b/o/p corners required")
    signs = np.array([-1.0, 1.0])
    output = []
    for mask in range(8):
        weight = np.ones((2, 2, 2))
        for bit in range(3):
            if mask & (1 << bit):
                shape = [1, 1, 1]
                shape[bit] = 2
                weight *= signs.reshape(shape)
        output.append(np.einsum("bop,bop...->...", weight, value) / 8)
    return np.stack(output)


def units(logits, normalizer):
    """Compute separate native score/odds/mass/entropy units from primitive logits."""
    from scipy.special import logsumexp

    z = np.asarray(logits, dtype=np.float64)
    if z.shape[-1] != 10 or not np.isfinite(z).all():
        raise ValueError("Ten finite digit logits required")
    local = logsumexp(z, axis=-1)
    p = np.exp(z - local[..., None])
    return np.stack(
        [
            p @ np.linspace(0, 1, 10),
            logsumexp(z[..., 5:], axis=-1) - logsumexp(z[..., :5], axis=-1),
            np.exp(local - np.asarray(normalizer)),
            -np.sum(p * (z - local[..., None]), axis=-1),
        ],
        axis=-1,
    )


def native_batch(observer, row, suffixes, alphabet, route, layer=17, *, noop=False):
    """Run a fixed native batch while changing only suffix attention permissions.

    All body computations are retained. Cuts act at every block and every suffix
    position so an earlier suffix token cannot bypass a forbidden body group.
    """
    import torch

    from grain_research.vector_mechanisms.direct_traits import native_readout

    prompts = [row["input_ids"] + list(s) for s in suffixes]
    if noop and route != 7:
        raise ValueError("Unmodified attention is only a full-access no-op control")
    lengths = [len(p) for p in prompts]
    batch, width = len(prompts), max(lengths)
    if not batch or width > observer.max_tokens:
        raise ValueError("Complete native execution support required")
    device = next(observer.model.parameters()).device
    ids = torch.zeros((batch, width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, prompt in enumerate(prompts):
        ids[index, : len(prompt)] = torch.tensor(prompt, device=device)
        mask[index, : len(prompt)] = 1
        mask[index, row["padding"]] = 0
    expected = np.stack([permissions(row, len(s), width, 7) for s in suffixes])
    desired = np.stack([permissions(row, len(s), width, route) for s in suffixes])
    desired_tensor = torch.tensor(desired[:, None], device=device)
    saved, calls, hooks = {}, [], []
    current, body = row["current"], len(row["input_ids"])

    def mask_hook(index):
        def apply(_module, _args, kwargs):
            actual = kwargs["attention_mask"]
            calls.append(index)
            if index == 0:
                if not np.array_equal((actual[:, 0] == 0).cpu().numpy(), expected):
                    raise ValueError("Unexpected original causal/padding permissions")
                saved["base_mask"] = actual.clone()
            elif not torch.equal(actual, saved["base_mask"]):
                raise ValueError("A decoder block received an unexpected base mask")
            changed = actual.clone().masked_fill(~desired_tensor, float("-inf"))
            if not torch.equal(changed == 0, desired_tensor):
                raise ValueError("Actual changed permissions differ from planned cut")
            kwargs["attention_mask"] = actual if noop else changed
            return _args, kwargs

        return apply

    def key_hook(_module, _args, output):
        saved["k_raw"] = output[:, current].float().cpu().numpy().copy()

    def value_hook(_module, _args, output):
        saved["v"] = output[:, current].float().cpu().numpy().copy()

    def final_body_hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        saved["body_final"] = hidden[0, :body].float().cpu().numpy().copy()

    try:
        for index, block in enumerate(observer.model.model.layers):
            hooks.append(
                block.self_attn.register_forward_pre_hook(
                    mask_hook(index), with_kwargs=True
                )
            )
        attention = observer.model.model.layers[layer].self_attn
        hooks.append(attention.k_proj.register_forward_hook(key_hook))
        hooks.append(attention.v_proj.register_forward_hook(value_hook))
        hooks.append(
            observer.model.model.layers[-1].register_forward_hook(final_body_hook)
        )
        endpoints = torch.tensor([n - 1 for n in lengths], device=device)
        with torch.inference_mode():
            result = observer.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=torch.arange(width, device=device)[None].expand(batch, -1),
                use_cache=False,
                logits_to_keep=endpoints,
                return_dict=True,
            )
        readings = [
            native_readout(result.logits[b, b].float().cpu().numpy(), alphabet)
            for b in range(batch)
        ]
    finally:
        for hook in hooks:
            hook.remove()
    observer.calls += 1
    if calls != list(range(len(observer.model.model.layers))):
        raise ValueError("Every decoder block must perform the declared cut")
    hashes = [
        np.frombuffer(hashlib.sha256(s.tobytes()).digest(), dtype=np.uint8)
        for s in desired
    ]
    return readings, {
        "k_raw": saved["k_raw"],
        "v": saved["v"],
        "body_final": saved["body_final"],
        "support_hash": np.stack(hashes),
        "blocks": np.asarray(calls),
        "execution_length": np.asarray(width),
        "endpoints": np.asarray(lengths) - 1,
    }
