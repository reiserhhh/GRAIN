#!/usr/bin/env python3
"""Exact context-prefix grouping and branched cached native/PQ reads."""

from __future__ import annotations

import copy

import numpy as np


def context_groups(design):
    """Share only an identical complete context before every question-region token."""
    groups = {}
    for d in design:
        ids = np.asarray(d["input_ids"])
        mask = np.asarray(d["mask"], bool)
        if ids.ndim != 1 or ids.shape != mask.shape or not mask.any() or mask[-1]:
            raise ValueError("Invalid prefix/question partition")
        boundary = int(np.flatnonzero(mask)[0])
        if boundary < 1:
            raise ValueError("Nonempty context required")
        groups.setdefault(tuple(ids[:boundary].tolist()), []).append(d)
    return [(list(prefix), rows) for prefix, rows in groups.items()]


def capture_uncached(observer, design, block=16):
    """Record only P/Q in a full pass, matching cached output/copy overhead."""
    import torch

    from .direct_traits import native_readout

    device = next(observer.model.parameters()).device
    saved = {}
    mask = np.asarray(design["mask"], bool)

    def hook(_module, _inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        saved["P"] = state[0, mask].detach().float().cpu().numpy().copy()
        saved["Q"] = state[0, -1].detach().float().cpu().numpy().copy()

    handle = observer.model.model.layers[block].register_forward_hook(hook)
    observer.calls += 1
    try:
        with torch.no_grad():
            ids = torch.tensor([design["input_ids"]], device=device)
            result = observer.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
            )
        return saved, native_readout(
            result.logits[0, -1].detach().float().cpu().numpy(), design["alphabet"]
        )
    finally:
        handle.remove()


def capture_cached_suffix(observer, design, prefix, cache, block=16):
    """Clone the context cache per question and record exact shifted P/Q slots."""
    import torch

    from .direct_traits import native_readout

    ids, mask = design["input_ids"], np.asarray(design["mask"], bool)
    n = len(prefix)
    if ids[:n] != prefix or mask[:n].any() or n >= len(ids):
        raise ValueError("Cache prefix must end before the question")
    if cache.get_seq_length() != n:
        raise ValueError("Unexpected original cache length")
    cloned = copy.deepcopy(cache)
    device = next(observer.model.parameters()).device
    saved = {}

    def hook(_module, _inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        saved["P"] = state[0, mask[n:]].detach().float().cpu().numpy().copy()
        saved["Q"] = state[0, -1].detach().float().cpu().numpy().copy()

    handle = observer.model.model.layers[block].register_forward_hook(hook)
    observer.calls += 1
    try:
        with torch.no_grad():
            result = observer.model(
                input_ids=torch.tensor([ids[n:]], device=device),
                attention_mask=torch.ones(
                    (1, len(ids)), dtype=torch.long, device=device
                ),
                position_ids=torch.arange(n, len(ids), device=device)[None, :],
                past_key_values=cloned,
                use_cache=True,
                logits_to_keep=1,
            )
        native = native_readout(
            result.logits[0, -1].detach().float().cpu().numpy(), design["alphabet"]
        )
        if cache.get_seq_length() != n or cloned.get_seq_length() != len(ids):
            raise RuntimeError("A suffix mutated its shared parent or failed to append")
        return saved, native
    finally:
        handle.remove()
