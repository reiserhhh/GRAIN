#!/usr/bin/env python3
"""Complete role/content observation laws, retaining writer and reader products."""

from __future__ import annotations

import numpy as np

CHARACTERS = ("mean", "content", "owner", "joint")
QUERY_CHANNELS = (
    "common_weight_common_value",
    "common_weight_specific_value",
    "specific_weight_common_value",
    "specific_weight_specific_value",
)
JOINT_PRODUCTS = (
    "value_joint",
    "content_weight_owner_value",
    "owner_weight_content_value",
    "weight_joint",
)
BASIS = np.array(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
    dtype=float,
)


def coefficients(corners):
    """Use the full Z2 x Z2 character basis on [high/A, low/A, high/B, low/B]."""
    x = np.asarray(corners, float)
    if x.ndim < 1 or x.shape[0] != 4 or not np.isfinite(x).all():
        raise ValueError("Four finite ordered corners required")
    return np.tensordot(BASIS / 4, x, axes=(1, 0))


def reconstruct(parts):
    """Invert the complete character transform without discarding main effects."""
    a = np.asarray(parts, float)
    if a.ndim < 1 or a.shape[0] != 4 or not np.isfinite(a).all():
        raise ValueError("Four finite character coefficients required")
    return np.tensordot(BASIS.T, a, axes=(1, 0))


def primitives(keys, values, queries, directions):
    """Read normalized token weights and directional values with frozen GQA units.

    Output shape is corner, token, target, query-head. This reads a declared
    attention function, not a final decoder prediction or attention explanation.
    """
    k, v, q, d = (np.asarray(a, float) for a in (keys, values, queries, directions))
    if (
        k.ndim != 4
        or k.shape[0] != 4
        or k.shape[1] == 0
        or k.shape[2] == 0
        or v.shape != k.shape
        or q.ndim != 3
        or q.shape[0] == 0
        or q.shape[1] == 0
        or d.shape != q.shape
        or q.shape[-1] != k.shape[-1]
        or q.shape[1] % k.shape[2]
        or not all(np.isfinite(a).all() for a in (k, v, q, d))
    ):
        raise ValueError(
            "Four compatible finite K/V corners and source readers required"
        )
    heads = np.arange(q.shape[1]) // (q.shape[1] // k.shape[2])
    logits = np.einsum("aind,qnd->aiqn", k[:, :, heads], q)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    content = np.einsum("aind,qnd->aiqn", v[:, :, heads], d)
    return weights, content


def checked(weights, values):
    """Require actual probability weights for all four aligned token supports."""
    p, c = np.asarray(weights, float), np.asarray(values, float)
    if (
        p.ndim != 4
        or p.shape[0] != 4
        or min(p.shape) == 0
        or c.shape != p.shape
        or not np.isfinite(p).all()
        or not np.isfinite(c).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Aligned finite corner/token/target/head arrays required")
    return p, c


def product_law(weights, values):
    """Return every character product: a_j = sum_r <p_r,c_(r xor j)>.

    Terms are indexed by output character, weight character, target. Four terms
    of the joint output distinguish carried value interaction from reader
    multiplication of separate content/owner changes. Signed terms may cancel.
    """
    p, c = checked(weights, values)
    ph, ch = coefficients(p), coefficients(c)
    terms = np.stack(
        [
            np.stack([(ph[r] * ch[r ^ j]).sum(axis=(0, 2)) for r in range(4)])
            for j in range(4)
        ]
    )
    return terms


def query_product_law(weights, values):
    """Retain common/specific weight and covector products over a declared battery.

    Common means the arithmetic mean of normalized probabilities or directional
    values across the supplied targets, not a universal psychological factor.
    The specific terms are signed algebraic differences, not probability laws.
    """
    p, c = checked(weights, values)
    common_p = np.broadcast_to(p.mean(axis=2, keepdims=True), p.shape)
    common_c = np.broadcast_to(c.mean(axis=2, keepdims=True), c.shape)
    fields = (
        (common_p, common_c),
        (common_p, c - common_c),
        (p - common_p, common_c),
        (p - common_p, c - common_c),
    )
    out = []
    for a, b in fields:
        ah, bh = coefficients(a), coefficients(b)
        out.append(
            np.stack(
                [
                    np.stack([(ah[r] * bh[r ^ j]).sum(axis=(0, 2)) for r in range(4)])
                    for j in range(4)
                ]
            )
        )
    return np.stack(out)


def author_score(observation_coefficients, probabilities):
    """Compose an explicit positive author evidence law without attention mass tilt."""
    a, p = np.asarray(observation_coefficients, float), np.asarray(probabilities, float)
    if (
        a.ndim < 1
        or a.shape[0] != 4
        or not np.isfinite(a).all()
        or p.shape != (4,)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError(
            "Complete observation coefficients and a positive law required"
        )
    return np.tensordot(BASIS @ p, a, axes=(0, 0))
