#!/usr/bin/env python3
"""Actual event-to-event/relay permission cuts and recursive-collector accounts."""

from __future__ import annotations

import hashlib

import numpy as np

ARMS = ((0, 0), (1, 0), (0, 1), (1, 1))
COMPONENTS = ("collector", "event_context", "common_relay", "route_interaction")


def permission_mask(row, cross_events, relay):
    """Cut event contextualization without cutting the current carrier's access."""
    if cross_events not in (0, 1) or relay not in (0, 1):
        raise ValueError("Binary declared permissions required")
    length = len(row["input_ids"])
    events = [np.asarray(e, int) for e in row["events"]]
    p, r, c, pad = (
        np.asarray(row[k], int) for k in ("prefix", "relay", "current", "padding")
    )
    all_positions = np.concatenate([p, *events, r, c, pad])
    if (
        any(not len(e) for e in [p, *events, r, c])
        or not np.array_equal(np.sort(all_positions), np.arange(length))
        or max(np.concatenate(events)) >= min(r)
        or max(r) >= min(c)
        or max(p) >= min(np.concatenate(events))
    ):
        raise ValueError(
            "Ordered complete disjoint prefix/event/relay/carrier partition required"
        )
    allowed = np.tri(length, dtype=bool)
    allowed[:, pad] = False
    event_tokens = np.concatenate(events)
    if not cross_events:
        for event in events:
            other = np.setdiff1d(event_tokens, event)
            allowed[np.ix_(event, other)] = False
    if not relay:
        allowed[np.ix_(r, event_tokens)] = False
    if not allowed.any(1).all():
        raise ValueError("Empty attention support")
    return allowed


def capture(observer, row):
    """Scope surgery to actual blocks0--16; return full final K/V for probes."""
    import torch

    from grain_research.vector_mechanisms.prefix import capture_prefix

    c, r = row["arm"]
    allowed = permission_mask(row, c, r)
    handles, calls = [], []

    def hook(index):
        def apply(_module, args, kwargs):
            original = kwargs.get("attention_mask")
            if original is None or original.shape[-2:] != allowed.shape:
                raise ValueError("Stock explicit causal mask required")
            denied = torch.as_tensor(~allowed, device=original.device)[None, None]
            actual = original.clone().masked_fill(
                denied, torch.finfo(original.dtype).min
            )
            if not torch.equal(actual[0, 0] == 0, ~denied[0, 0]):
                raise ValueError("Actual permissions differ from the declared graph")
            calls.append(index)
            return args, dict(kwargs, attention_mask=actual)

        return apply

    try:
        for layer in range(17):
            handles.append(
                observer.model.model.layers[layer].self_attn.register_forward_pre_hook(
                    hook(layer), with_kwargs=True
                )
            )
        raw, trace = capture_prefix(observer, row["input_ids"])
    finally:
        for handle in handles:
            handle.remove()
    if calls != list(range(17)):
        raise RuntimeError("Incomplete actual route intervention")
    return raw, {
        **trace,
        "masked_blocks": calls,
        "arm": row["arm"],
        "permission_sha256": hashlib.sha256(allowed.tobytes()).hexdigest(),
        "allowed_edges": int(allowed.sum()),
    }


def route_account(corners):
    """Resolve four actual permission arms, preserving signed interaction."""
    f = np.asarray(corners, float)
    if f.shape[0] != 4 or not np.isfinite(f).all():
        raise ValueError("Complete finite [00,10,01,11] field required")
    return np.stack((f[0], f[1] - f[0], f[2] - f[0], f[3] - f[1] - f[2] + f[0]))


def referenced_probe(raw, positions):
    """Keep aligned primitives for a structural-invariance check, never fitting."""
    index = np.asarray(positions, int)
    return np.stack((raw["k"][index], raw["v"][index]))
