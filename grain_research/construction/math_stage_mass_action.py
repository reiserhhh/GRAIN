#!/usr/bin/env python3
"""Fixed-stage interventions separate propagated and later history access."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_recursive_collector as C
from grain_research.construction import math_supported_attention_safe as S


def arm_indices():
    """Keep baseline then cue/alignment/sign/route in a fixed 49-arm order."""
    return np.array(list(np.ndindex(6, 2, 2, 2)), dtype=int)


def mass_pairs(arms):
    """Build early/late measures; do not change dose, marks or token support."""
    a = np.asarray(arms, float)
    if a.ndim != 3 or a.shape[1] != 25 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("All25 finite nonnegative HM122 mass arms required")
    early, late = [a[:, 0]], [a[:, 0]]
    for q, align, sign, route in arm_indices():
        edited = a[:, 1 + q * 4 + align * 2 + sign]
        early.append(edited if route == 0 else a[:, 0])
        late.append(a[:, 0] if route == 0 else edited)
    return np.stack(early, 1), np.stack(late, 1)


def construct(k, v, early, late, positions, origin, seeds, cos, sin, weights, cfg, cut):
    """Use edited mass on one fixed stage, propagating all current states."""
    import torch
    from torch.nn import functional as f

    blocks = len(weights) - 1
    if not 0 < cut < blocks or k.shape != v.shape or k.shape[1] != blocks + 1:
        raise ValueError("A fixed interior stage and complete history maps required")
    if early.shape != late.shape or early.shape != (k.shape[0], k.shape[2]):
        raise ValueError("Early/late measures must address the same history")
    positions = torch.as_tensor(positions, dtype=torch.long, device=cos.device)
    if bool((positions < 0).any()) or bool((positions >= origin).any()):
        raise ValueError("History positions must precede current tokens")
    batch, focuses, length = k.shape[0], seeds.shape[0], seeds.shape[1]
    h = seeds[None].expand(batch, -1, -1, -1).reshape(batch * focuses, length, -1)
    for layer in range(blocks):
        measure = early if layer < cut else late
        h = S.block(
            h,
            k[:, layer].repeat_interleave(focuses, 0),
            v[:, layer].repeat_interleave(focuses, 0),
            measure.repeat_interleave(focuses, 0),
            cos[positions],
            sin[positions],
            cos[origin:],
            sin[origin:],
            weights[layer],
            cfg,
        )
    x = C.rms_norm(h, weights[-1]["input_layernorm.weight"], cfg["rms_norm_eps"])
    dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    result = []
    for name in ("k", "v"):
        a = f.linear(
            x,
            weights[-1][f"self_attn.{name}_proj.weight"],
            weights[-1][f"self_attn.{name}_proj.bias"],
        ).reshape(batch * focuses, length, cfg["num_key_value_heads"], dim)
        if name == "k":
            a = C.rotary(a, cos[origin:], sin[origin:])
        if not bool(torch.isfinite(a).all()):
            raise FloatingPointError(
                "Preserve nonfinite staged result, no precision rescue"
            )
        result.append(a.reshape(batch, focuses, length, -1, dim))
    return tuple(result)


def route_readings(readings):
    """Recover baseline+24 arms for each fixed route, not selected best layers."""
    r = np.asarray(readings, float)
    if r.shape != (288, 49, 2, 2, 6) or not np.isfinite(r).all():
        raise ValueError("All288 histories and49 fixed arms required")
    shaped = r[:, 1:].reshape(288, 6, 2, 2, 2, 2, 2, 6)
    return tuple(
        np.concatenate(
            (r[:, :1], shaped[:, :, :, :, route].reshape(288, 24, 2, 2, 6)), 1
        )
        for route in range(2)
    )


def finite_account(base, early, late, both):
    """Exact two-switch Mobius account; do not label it unique mediation."""
    b, e, l, t = [np.asarray(x, float) for x in (base, early, late, both)]
    if len({x.shape for x in (b, e, l, t)}) != 1:
        raise ValueError("Every effect must share its observable and input address")
    if not all(np.isfinite(x).all() for x in (b, e, l, t)):
        raise ValueError("Finite effects required")
    return {"early": e - b, "late": l - b, "interaction": t - e - l + b, "total": t - b}


def cosine(x, y):
    """Return an unavailable direction explicitly rather than inventing zero."""
    a, b = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return None if n <= 1e-12 else float(np.dot(a, b) / n)
