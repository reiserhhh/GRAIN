#!/usr/bin/env python3
"""Conditional comment quartets and role-preserving retained-state references."""

from __future__ import annotations

import numpy as np


def partition(groups, p_mask):
    """Return disjoint contiguous bodies and template gaps, leaving P/Q untouched."""
    p = np.asarray(p_mask, bool)
    g = [np.asarray(x, int) for x in groups]
    if p.ndim != 1 or not p.any() or p[-1] or not g:
        raise ValueError("Nonfinal query mask and body groups required")
    stop = int(np.flatnonzero(p)[0])
    if not p[stop:-1].all():
        raise ValueError("Expected contiguous P followed by Q")
    previous, gaps = 0, []
    for x in g:
        if not len(x) or x[0] < previous or x[-1] >= stop or np.any(np.diff(x) != 1):
            raise ValueError("Ordered nonoverlapping contiguous bodies required")
        gaps.append(np.arange(previous, x[0]))
        previous = int(x[-1]) + 1
    gaps.append(np.arange(previous, stop))
    if any(not len(x) for x in gaps):
        raise ValueError("Each declared template role must be represented")
    body, rest = np.concatenate(g), np.concatenate(gaps)
    if len(set(np.concatenate([body, rest]))) != stop:
        raise ValueError("Body and retained states must partition pre-query context")
    return g, gaps, body, rest


def resample_role(values, length):
    """Linearly sample a donor role on normalized positions; equal grids are exact."""
    a = np.asarray(values)
    if a.ndim != 2 or not len(a) or length < 1 or not np.isfinite(a).all():
        raise ValueError("Finite nonempty role matrix required")
    if length == len(a):
        return a.copy()
    x = np.linspace(0, len(a) - 1, length)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(a) - 1)
    w = x - lo
    return a[lo] * (1 - w[:, None]) + a[hi] * w[:, None]


def reference_state(target, groups, p_mask, donors):
    """Replace only M/R by donor role averages; target values survive solely in P/Q."""
    h = np.asarray(target)
    g, gaps, _, _ = partition(groups, p_mask)
    if h.ndim != 2 or len(h) != len(p_mask) or not np.isfinite(h).all() or not donors:
        raise ValueError("Target layout and reference donors required")
    ref = h.copy()
    donor_parts = []
    interpolation_count = 0
    for values, dg, dp in donors:
        gg, rr, _, _ = partition(dg, dp)
        if len(gg) != len(g) or values.shape[1] != h.shape[1]:
            raise ValueError("Matched comment roles and hidden dimensions required")
        donor_parts.append((np.asarray(values, float), gg, rr))
    for j, positions in enumerate(g):
        ref[positions] = np.mean([a[gg[j]].mean(0) for a, gg, _ in donor_parts], axis=0)
    for j, positions in enumerate(gaps):
        ref[positions] = np.mean(
            [resample_role(a[rr[j]], len(positions)) for a, _, rr in donor_parts],
            axis=0,
        )
        interpolation_count += sum(
            len(rr[j]) != len(positions) for _, _, rr in donor_parts
        )
    return ref, interpolation_count


def score(logits, values):
    """Compute the original answer-alphabet softmax expectation in float64."""
    z, v = np.asarray(logits, float), np.asarray(values, float)
    if z.ndim != 1 or z.shape != v.shape or not np.isfinite([z, v]).all():
        raise ValueError("Matched finite answer vectors required")
    p = np.exp(z - z.max())
    return float(p @ v / p.sum())


def joint_account(z0, zi, zj, zij, values):
    """Split score interaction into nonadditive logits and output-link curvature."""
    a = np.asarray([z0, zi, zj, zij], float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Four finite paired logit endpoints required")
    s0, si, sj, sij = [score(x, values) for x in a]
    sadd = score(a[1] + a[2] - a[0], values)
    mixed = a[3] - a[1] - a[2] + a[0]
    delta = a[3] - a[0]
    return {
        "f0": s0,
        "fi": si,
        "fj": sj,
        "fij": sij,
        "pair_action": sij - s0,
        "effect_i": si - s0,
        "effect_j": sj - s0,
        "interaction": sij - si - sj + s0,
        "internal_score": sij - sadd,
        "link_curvature": sadd - si - sj + s0,
        "logit_interaction_rms": float(np.sqrt(np.mean((mixed - mixed.mean()) ** 2))),
        "logit_pair_action_rms": float(np.sqrt(np.mean((delta - delta.mean()) ** 2))),
        "i_sign_flip": bool((si - s0) * (sij - sj) < 0),
        "j_sign_flip": bool((sj - s0) * (sij - si) < 0),
    }
