#!/usr/bin/env python3
"""Supported history measures and mass-action marks on fixed rotary positions.

This is a chosen instrument path, not a natural text trajectory. Positions are
part of each atom; duplicating an atom must not move current-token positions.
"""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_marked_response as M


def pack(low_k, high_k, low_v, high_v, low_mass, high_mass):
    """Discard zero-mass representatives and retain two supported atom families."""
    import torch

    if (
        low_k.shape != high_k.shape
        or low_k.shape != low_v.shape
        or high_k.shape != high_v.shape
        or low_k.ndim != 5
        or low_mass.shape != high_mass.shape
        or low_mass.shape != (low_k.shape[0], low_k.shape[2])
        or bool((low_mass < 0).any())
        or bool((high_mass < 0).any())
    ):
        raise ValueError("Paired layer histories and nonnegative support required")
    values = []
    for low, high in ((low_k, high_k), (low_v, high_v)):
        a = torch.where(low_mass[:, None, :, None, None] > 0, low, 0)
        b = torch.where(high_mass[:, None, :, None, None] > 0, high, 0)
        values.append(torch.cat((a, b), dim=2))
    zero = torch.zeros_like(low_mass)
    start = torch.cat((low_mass, zero), dim=1)
    change = torch.cat((-low_mass, high_mass), dim=1)
    return (*values, start, change)


def construct(k, v, mass, positions, current_origin, seeds, cos, sin, weights, config):
    """Run immutable current arithmetic on explicitly positioned measure atoms."""
    import torch

    positions = torch.as_tensor(positions, dtype=torch.long, device=cos.device)
    if (
        positions.ndim != 1
        or len(positions) != k.shape[2]
        or bool((positions < 0).any())
        or bool((positions >= current_origin).any())
        or cos.shape[0] != current_origin + seeds.shape[1]
    ):
        raise ValueError("History atoms retain old positions before current origin")
    c = torch.cat((cos[positions], cos[current_origin:]), dim=0)
    s = torch.cat((sin[positions], sin[current_origin:]), dim=0)
    return M.construct(k, v, mass, seeds, c, s, weights, config)


def moments(action, marks, visible, sides):
    """Push signed mass action to marks, with side-preserving wrong-slot control."""
    a, g, mask = (
        np.asarray(action, float),
        np.asarray(marks, float),
        np.asarray(visible, bool),
    )
    sides = np.asarray(sides, int)
    if (
        a.ndim != 4
        or g.shape != (a.shape[0], a.shape[-1], 4)
        or mask.shape != g.shape[:2]
        or sides.shape != (a.shape[-1],)
        or not np.isfinite(a).all()
        or not np.isfinite(g).all()
        or not mask.any(1).all()
    ):
        raise ValueError("Finite actions and four addressed endpoint marks required")
    average = (g * mask[..., None]).sum(1) / mask.sum(1)[:, None]
    tagged = a * mask[:, None, None]
    first = np.einsum("pmqi,pij->pmqj", tagged, g)
    connected = first - tagged.sum(-1)[..., None] * average[:, None, None]
    wrong = np.zeros_like(g)
    for p in range(len(g)):
        for side in np.unique(sides):
            ix = np.flatnonzero(mask[p] & (sides == side))
            wrong[p, ix] = np.roll(g[p, ix], len(ix) // 2, axis=0)
    return {
        "mass": a.sum(-1),
        "body_mass": tagged.sum(-1),
        "untagged_mass": (a - tagged).sum(-1),
        "first": first,
        "connected": connected,
        "misaligned_connected": np.einsum(
            "pmqi,pij->pmqj", tagged, wrong - average[:, None]
        ),
        "total_variation": abs(a).sum(-1),
    }
