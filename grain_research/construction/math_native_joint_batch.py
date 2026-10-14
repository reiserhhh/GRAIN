#!/usr/bin/env python3
"""Same-execution-shape native observation; no decoder readout fitting."""

from __future__ import annotations

import hashlib

import numpy as np

from grain_research.construction.math_native_joint_value import edit_values


def native_batch(observer, row, suffixes, alphabet, delta, layer=17):
    """Read all target suffixes together, retaining native endpoint and pad masks.

    The right-padding is excluded from attention. Each target's final logits
    are selected at its own real endpoint, never at the common padding endpoint.
    Shared current arrays are observed, not assumed equal to a shorter forward.
    """
    import torch

    from grain_research.vector_mechanisms.direct_traits import native_readout
    from grain_research.vector_mechanisms.prefix import rotate_half

    prompts = [row["input_ids"] + list(s) for s in suffixes]
    lengths = [len(p) for p in prompts]
    length, batch = max(lengths), len(prompts)
    if length > observer.max_tokens or not batch:
        raise ValueError("Full native prompts required within the fixed budget")
    device = next(observer.model.parameters()).device
    ids = torch.zeros((batch, length), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for b, prompt in enumerate(prompts):
        ids[b, : len(prompt)] = torch.tensor(prompt, device=device)
        mask[b, : len(prompt)] = 1
        mask[b, row["padding"]] = 0
    current = row["current"]
    attention = observer.model.model.layers[layer].self_attn
    dim, saved, hooks, calls = attention.head_dim, {}, [], []
    width = attention.v_proj.out_features
    if delta is not None and np.asarray(delta).shape != (
        len(current),
        width // dim,
        dim,
    ):
        raise ValueError("Actual physical value edit has wrong support")
    endpoints = torch.tensor([n - 1 for n in lengths], device=device)

    def mask_hook(index):
        def capture(_module, _args, kwargs):
            calls.append(index)
            actual = kwargs["attention_mask"]
            if index == 0:
                support = (actual[:, 0] == 0).cpu().numpy()
                hashes, whole = [], []
                for b, n in enumerate(lengths):
                    expected = np.tri(length, dtype=bool)
                    expected[:, row["padding"]] = False
                    expected[:, n:] = False
                    if not np.array_equal(expected, support[b]):
                        raise ValueError(
                            "Actual batched causal/padding permissions changed"
                        )
                    hashes.append(
                        np.frombuffer(
                            hashlib.sha256(support[b, :n, :n].tobytes()).digest(),
                            dtype=np.uint8,
                        )
                    )
                    whole.append(
                        np.frombuffer(
                            hashlib.sha256(support[b].tobytes()).digest(),
                            dtype=np.uint8,
                        )
                    )
                saved["support_hash"] = np.stack(hashes)
                saved["execution_support_hash"] = np.stack(whole)
            if index == layer:
                cos, sin = kwargs["position_embeddings"]
                saved["cos"] = cos[:, current].float().cpu().numpy().copy()
                saved["sin"] = sin[:, current].float().cpu().numpy().copy()

        return capture

    def key_hook(_module, _args, output):
        saved["raw_key"] = output[:, current].float().cpu().numpy().copy()

    def value_hook(_module, _args, output):
        saved["before"] = output[:, current].float().cpu().numpy().copy()
        if delta is not None:
            output = torch.cat(
                [
                    edit_values(
                        output[b : b + 1],
                        current,
                        np.asarray(delta).reshape(len(current), -1),
                    )
                    for b in range(batch)
                ]
            )
        saved["after"] = output[:, current].float().cpu().numpy().copy()
        return output

    try:
        for i, block in enumerate(observer.model.model.layers):
            hooks.append(
                block.self_attn.register_forward_pre_hook(
                    mask_hook(i), with_kwargs=True
                )
            )
        hooks.append(attention.k_proj.register_forward_hook(key_hook))
        hooks.append(attention.v_proj.register_forward_hook(value_hook))
        with torch.inference_mode():
            result = observer.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=torch.arange(length, device=device)[None].expand(
                    batch, -1
                ),
                use_cache=False,
                logits_to_keep=endpoints,
                return_dict=True,
            )
        scores = [
            native_readout(result.logits[b, b].float().cpu().numpy(), alphabet)
            for b in range(batch)
        ]
    finally:
        for hook in hooks:
            hook.remove()
    observer.calls += 1
    if calls != list(range(len(observer.model.model.layers))):
        raise RuntimeError("Incomplete batched native execution trace")
    key = saved["raw_key"].reshape(batch, len(current), -1, dim)
    cos, sin = saved["cos"][:, :, None], saved["sin"][:, :, None]
    if cos.shape[0] == 1:
        cos, sin = np.broadcast_to(cos, key.shape), np.broadcast_to(sin, key.shape)
    return scores, {
        "k": key * cos + rotate_half(key) * sin,
        "before": saved["before"].reshape(key.shape),
        "after": saved["after"].reshape(key.shape),
        "support_hash": saved["support_hash"],
        "execution_support_hash": saved["execution_support_hash"],
        "masked_blocks": np.tile(np.arange(len(calls)), (batch, 1)),
        "execution_length": np.full(batch, length),
        "endpoint": np.asarray(lengths) - 1,
    }
