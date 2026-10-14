#!/usr/bin/env python3
"""Source-only physical value actions and actor-differential observations."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_action_expression import actor_fields
from grain_research.construction.math_conditional_chart import decompose

ARMS = ("baseline", "source_odd", "source_even", "cyclic_family", "random")


def head_match(value, reference):
    """Match each physical head norm without using target/native responses."""
    x, r = np.asarray(value, float), np.asarray(reference, float)
    if x.shape != r.shape or x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("Finite equal token/head/dimension arrays required")
    xn = np.linalg.norm(x, axis=(0, 2), keepdims=True)
    rn = np.linalg.norm(r, axis=(0, 2), keepdims=True)
    if np.any(xn <= 1e-12) or not np.isfinite(r).all():
        raise ValueError("Unavailable head direction cannot be filled with zero")
    return x * rn / xn


def source_actions(values, seed=116):
    """Use only original behavior/wording0 and construct fixed matched controls."""
    x = np.asarray(values, float)
    if x.ndim != 9 or x.shape[:6] != (3, 2, 6, 2, 2, 2):
        raise ValueError(
            "All HM115 fields with physical token/head coordinates required"
        )
    # Source slicing occurs before any operation on expression/application fields.
    selected = x[0, 0].reshape(6, 2, 2, 2, -1)
    f = actor_fields(selected)
    shape = (6, 2, *x.shape[6:])
    d, s = f["odd"].reshape(shape), f["even"].reshape(shape)
    result = {
        "source_odd": d,
        "source_even": np.empty_like(d),
        "cyclic_family": np.empty_like(d),
        "random": np.empty_like(d),
    }
    for family in range(6):
        for c in range(2):
            ref = d[family, c]
            result["source_even"][family, c] = head_match(s[family, c], ref)
            result["cyclic_family"][family, c] = head_match(d[(family + 1) % 6, c], ref)
            rng = np.random.default_rng(np.random.SeedSequence([seed, family, c]))
            result["random"][family, c] = head_match(rng.normal(size=ref.shape), ref)
    return result


def effect_accounts(native):
    """Keep all characters plus common/differential actor actions in each unit."""
    x = np.asarray(native, float)
    if x.shape != (5, 2, 6, 2, 2, 2, 2, 6, 4) or not np.isfinite(x).all():
        raise ValueError("Complete arm/wording/family/c/b/o/person/query/unit required")
    beta = np.empty((5, 2, 6, 8, 2, 6, 4))
    for arm in range(5):
        for w in range(2):
            for f in range(6):
                beta[arm, w, f] = decompose(x[arm, w, f].reshape(8, -1)).reshape(
                    8, 2, 6, 4
                )
    effect = beta[0:1] - beta
    return {
        "native": beta,
        "effects": effect,
        "actor_difference": (effect[..., 0, :, :] - effect[..., 1, :, :]) / 2,
        "actor_common": (effect[..., 0, :, :] + effect[..., 1, :, :]) / 2,
    }


def author_difference(cells, amplitude=0.4):
    """Direct positive-law integral with mass1/6, not a real human score."""
    x = np.asarray(cells, float)
    if x.shape[:3] != (2, 2, 2) or not 0 < amplitude <= 1:
        raise ValueError("Complete c/b/o cells and legal amplitude required")
    total = np.zeros(x.shape[3:])
    for c in range(2):
        for b in range(2):
            for o in range(2):
                chi = (2 * b - 1) * (2 * o - 1)
                total += amplitude * chi * x[c, b, o] / 24
    return total
