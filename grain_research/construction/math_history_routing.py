#!/usr/bin/env python3
"""Finite causal-route interventions and signed author observation accounts."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from grain_research.construction.math_cold_evidence import rotate


def route_mask(length, history, relay, current, padding, direct, indirect):
    """Preserve causality while cutting declared H->C and H->B permissions."""
    parts = [np.asarray(p, int) for p in (history, relay, current)]
    joined = np.concatenate(parts)
    if (
        length < 2
        or any(not len(p) for p in parts)
        or not np.array_equal(np.sort(joined), np.arange(length))
        or len(np.unique(joined)) != length
        or parts[0].max() >= parts[2].min()
    ):
        raise ValueError("Complete ordered, disjoint H/B/C token partition required")
    pad = np.asarray(padding, int)
    if not np.isin(pad, parts[0]).all():
        raise ValueError("Ignored padding must belong to H")
    allowed = np.tri(length, dtype=bool)
    allowed[:, pad] = False
    if not direct:
        allowed[np.ix_(parts[2], parts[0])] = False
    if not indirect:
        allowed[np.ix_(parts[1], parts[0])] = False
    if not allowed.any(1).all():
        raise ValueError("Route cut produced an empty softmax support")
    return allowed


def dependency_support(allowed, history, layers):
    """Propagate permitted dependencies, not actual nonlinear causal effects."""
    a = np.asarray(allowed, bool)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or layers < 0:
        raise ValueError("Square dependency graph and nonnegative depth required")
    reached = np.zeros(len(a), bool)
    reached[np.asarray(history, int)] = True
    for _ in range(layers):
        reached |= (a.astype(int) @ reached.astype(int)) > 0
    return reached


def mobius(corners):
    """Resolve all four route conditions without choosing an update order."""
    f = np.asarray(corners, float)
    if f.shape[0] != 4 or not np.isfinite(f).all():
        raise ValueError("Finite [neither,direct,relay,both] corners required")
    return np.stack((f[0], f[1] - f[0], f[2] - f[0], f[3] - f[1] - f[2] + f[0]))


def author_laws():
    """Return positive paired laws with equal occupancy and symmetric traffic."""
    d = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], float)
    return (1 + 0.5 * d) / 9, (1 - 0.5 * d) / 9


def author_contrast(field):
    """Evaluate the same-marginal author contrast on every retained channel."""
    f = np.asarray(field, float)
    if f.shape[:2] != (3, 3) or not np.isfinite(f).all():
        raise ValueError("Complete finite three-state observation kernel required")
    p, m = author_laws()
    return np.tensordot(p - m, f, axes=((0, 1), (0, 1)))


def read_current(k, v, endpoint, bundle):
    """Apply only frozen source query/covectors to current-token primitives."""
    k, v = np.asarray(k, float), np.asarray(v, float)
    directions = np.asarray(bundle["directions"], float)
    if k.shape != v.shape or k.ndim != 3 or not np.isfinite(k + v).all():
        raise ValueError("Finite current-token GQA key/value arrays required")
    heads = directions.shape[1]
    if heads % v.shape[1]:
        raise ValueError("Source/query GQA mismatch")
    ratio = heads // v.shape[1]
    values = np.repeat(v, ratio, axis=1)
    scores = np.empty((2, len(directions)))
    for target, direction in enumerate(directions):
        q = rotate(
            np.asarray(bundle["raw_queries"])[target],
            endpoint + bundle["gaps"][target],
            bundle["frequencies"],
            bundle["scaling"],
        )
        logits = np.stack([k[:, h // ratio] @ q[h] for h in range(heads)])
        content = np.einsum("ihd,hd->hi", values, direction)
        scores[:, target] = (
            content.mean(1).sum(),
            np.sum(softmax(logits, 1) * content),
        )
    return scores


def capture_routed_prefix(observer, row, direct, indirect):
    """Scope causal-mask surgery to blocks0--16 and stop before block17 attention."""
    import hashlib

    import torch

    from grain_research.vector_mechanisms.prefix import capture_prefix

    allowed = route_mask(
        len(row["input_ids"]),
        row["history"],
        row["relay"],
        row["current"],
        row["padding"],
        direct,
        indirect,
    )
    handles, calls = [], []

    def change(index):
        def hook(_module, args, kwargs):
            original = kwargs.get("attention_mask")
            if original is None or original.shape[-2:] != allowed.shape:
                raise ValueError("Expected stock explicit causal attention mask")
            denied = torch.as_tensor(~allowed, device=original.device)[None, None]
            updated = original.clone().masked_fill(
                denied, torch.finfo(original.dtype).min
            )
            if not torch.equal(updated[0, 0] == 0, ~denied[0, 0]):
                raise ValueError("Actual mask differs from declared causal permissions")
            kwargs = dict(kwargs, attention_mask=updated)
            calls.append(index)
            return args, kwargs

        return hook

    try:
        for index in range(17):
            handles.append(
                observer.model.model.layers[index].self_attn.register_forward_pre_hook(
                    change(index), with_kwargs=True
                )
            )
        raw, trace = capture_prefix(observer, row["input_ids"])
    finally:
        for handle in handles:
            handle.remove()
    if calls != list(range(17)):
        raise RuntimeError("Incomplete or out-of-order route interventions")
    current = np.asarray(row["current"], int)
    return {"k": raw["k"][current], "v": raw["v"][current]}, {
        **trace,
        "masked_blocks": calls,
        "direct": direct,
        "relay": indirect,
        "permission_sha256": hashlib.sha256(allowed.tobytes()).hexdigest(),
        "allowed_edges": int(allowed.sum()),
    }
