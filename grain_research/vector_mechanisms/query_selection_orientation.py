#!/usr/bin/env python3
"""Crossed query selection and direction, with an explicit conditional operator."""

import numpy as np

from .fixed_seed_attention import rotary
from .query_partition_geometry import partition
from .source_query_selection import key_actions


def coordinates(query_anchors, evidence):
    """Validate the pure evidence contract and render two anchors in recipient RoPE."""
    q = np.asarray(query_anchors, float)
    if q.ndim != 3 or len(q) != 2:
        raise ValueError("Exactly two source anchors required")
    diagonal = key_actions(q, evidence)
    e = {k: np.asarray(v, float) for k, v in evidence.items()}
    return (
        rotary(q, e["cos"][-1], e["sin"][-1]),
        rotary(e["k"], e["cos"][:, None], e["sin"][:, None]),
        rotary(e["dk"], e["cos"][None, :, None], e["sin"][None, :, None]),
        e,
        diagonal,
    )


def operator_actions(operator, rotated_directions):
    """Contract [selection,mode,factor,head,dim] with [direction,head,dim]."""
    c, q = np.asarray(operator, float), np.asarray(rotated_directions, float)
    if (
        c.ndim != 5
        or q.ndim != 3
        or c.shape[-2:] != q.shape[-2:]
        or not np.isfinite(c).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("Aligned finite operator and direction anchors required")
    return np.einsum("amfhk,bhk->abmf", c, q)


def crossed(query_anchors, evidence):
    """Compile all four paths and a vector-valued covariance operator.

    The operator is Cov_a(v,delta K)/sqrt(h), conditional on the selection
    anchor, but not the direction anchor. Off-diagonals are diagnostic paths.
    """
    q, k, dk, e, expected = coordinates(query_anchors, evidence)
    h, dim = q.shape[1:]
    groups, scale = h // k.shape[1], float(e["scale"])
    op = np.zeros((2, 2, len(dk), h, dim))
    direct = np.zeros((2, 2, 2, len(dk)))
    for a in range(2):
        for head in range(h):
            kv = head // groups
            logits = k[:, kv] @ q[a, head] * scale + e["mask"]
            allowed = np.isfinite(logits)
            p = np.exp(logits[allowed] - logits[allowed].max())
            p /= p.sum()
            v = e["v"][allowed, kv] @ e["g"][:, head].T
            d = dk[:, allowed, kv]
            vc = v - p @ v
            dc = d - np.einsum("t,ftk->fk", p, d)[:, None]
            op[a, :, :, head] = np.einsum("t,tm,ftk->mfk", p, vc, dc) * scale
            for b in range(2):
                ds = np.einsum("ftk,k->tf", d, q[b, head]) * scale
                m = partition(logits[allowed], np.column_stack((v, ds)))
                direct[a, b] += m["covariance"][:2, 2:]
    actions = operator_actions(op, q)
    np.testing.assert_allclose(actions, direct, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(
        np.stack((actions[0, 0], actions[1, 1])), expected, atol=1e-11, rtol=1e-10
    )
    return {"actions": actions, "operator": op, "anchors": q}


def path_score(query_anchors, evidence, selection, direction, factor, t):
    """Evaluate S_ab(t) with frozen Q/value/readout and one key-logit direction."""
    q, k, dk, e, _ = coordinates(query_anchors, evidence)
    if (
        selection not in (0, 1)
        or direction not in (0, 1)
        or factor not in range(len(dk))
    ):
        raise ValueError("Known crossed path and GRAIN factor required")
    output = np.zeros(2)
    groups, scale = q.shape[1] // k.shape[1], float(e["scale"])
    for h in range(q.shape[1]):
        kv = h // groups
        l = k[:, kv] @ q[selection, h] * scale + e["mask"]
        d = dk[factor, :, kv] @ q[direction, h] * scale
        value = e["v"][:, kv] @ e["g"][:, h].T
        output += partition(l + t * d, value)["mean"]
    return output


def decompose(actions):
    """Symmetric finite selection/direction allocation, with interaction separate."""
    a = np.asarray(actions, float)
    if a.shape[:2] != (2, 2) or not np.isfinite(a).all():
        raise ValueError("Finite two-by-two crossed responses required")
    selection = 0.5 * (a[1, 0] - a[0, 0] + a[1, 1] - a[0, 1])
    direction = 0.5 * (a[0, 1] - a[0, 0] + a[1, 1] - a[1, 0])
    delta = a[1, 1] - a[0, 0]
    interaction = a[1, 1] - a[1, 0] - a[0, 1] + a[0, 0]
    np.testing.assert_allclose(selection + direction, delta, atol=1e-12)
    return {
        "delta": delta,
        "selection": selection,
        "direction": direction,
        "interaction": interaction,
    }


def account(actions, target):
    """Signed projections of swap terms onto native targets and the total swap."""
    a, y = np.asarray(actions, float), np.asarray(target, float).ravel()
    d = decompose(a)
    if a[0, 0].size != len(y) or not np.isfinite(y).all():
        raise ValueError("Matched finite native target required")
    energy = float(y @ y)
    delta = d["delta"].ravel()
    swap_energy = float(delta @ delta)
    ratio = lambda n, den: float(n / den) if den else np.nan
    result = {"native_rms": float(np.sqrt(np.mean(y * y))), "actions": len(y)}
    for n, z in d.items():
        z = z.ravel()
        result[n + "_rms"] = float(np.sqrt(np.mean(z * z)))
        result[n + "_native_projection"] = ratio(z @ y, energy)
        result[n + "_swap_projection"] = ratio(z @ delta, swap_energy)
    s, r = d["selection"].ravel(), d["direction"].ravel()
    result["selection_energy_over_swap"] = ratio(s @ s, swap_energy)
    result["direction_energy_over_swap"] = ratio(r @ r, swap_energy)
    result["cross_energy_over_swap"] = ratio(2 * s @ r, swap_energy)
    result["closure_max"] = float(np.max(np.abs(s + r - delta)))
    return result
