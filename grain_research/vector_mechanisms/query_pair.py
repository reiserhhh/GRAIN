#!/usr/bin/env python3
"""Compose source-local baseline reading and source-matched comment actions."""

from __future__ import annotations

import numpy as np

from .complete_pair import ARMS, PAIRS, fit_complete, predict_complete
from .joint_readout import centered_logits, fit_joint


def fit_query_pair(blocks, phi, z0, increments, mixed, affine, values, penalty=0.01):
    """Fit five small heads using only one declared set of source authors."""
    z0, mixed = centered_logits(z0), centered_logits(mixed)
    n, candidates = z0.shape
    if mixed.shape != (n, 2, candidates):
        raise ValueError("Two source pair targets per author required")
    heads = {}
    for arm in ARMS:
        f = np.asarray(phi[arm], float)
        if f.ndim != 3 or f.shape[:2] != (n, 3):
            raise ValueError("Source-aligned three-comment features required")
        left = np.stack([f[:, i] for i, _ in PAIRS], 1).reshape(n * 2, -1)
        right = np.stack([f[:, j] for _, j in PAIRS], 1).reshape(n * 2, -1)
        heads[arm] = fit_joint(left, right, mixed.reshape(n * 2, -1), penalty)
    model = fit_complete(blocks, phi, z0, increments, heads, penalty, values)
    for key in ("intercept", "jm", "jr"):
        value = np.asarray(affine[key], float)
        if not np.isfinite(value).all():
            raise ValueError("Finite frozen baseline coefficients required")
        model["affine_" + key] = value.copy()
    if model["affine_intercept"].shape != (candidates,):
        raise ValueError("Affine alphabet differs from pair alphabet")
    for key, block in (("jm", "M"), ("jr", "R")):
        j = model["affine_" + key]
        if (
            j.ndim != 3
            or j.shape[0] != candidates
            or np.prod(j.shape[1:]) != blocks[block].shape[1]
        ):
            raise ValueError("Affine coordinate layout differs from pair inputs")
    return model


def scores(logits, values):
    """Convert arbitrary leading logit axes using one fixed candidate alphabet."""
    z, values = np.asarray(logits, float), np.asarray(values, float)
    if (
        z.shape[-1:] != values.shape
        or not np.isfinite(z).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("Finite logits and matching alphabet required")
    p = np.exp(z - z.max(-1, keepdims=True))
    return (p @ values) / p.sum(-1)


def predict_query_pair(blocks, phi, model):
    """Predict pair scores with no current-author endpoints or fitted normalization."""
    result = predict_complete(blocks, phi, model)
    z = np.asarray(model["affine_intercept"])[None].copy()
    for key, block in (("jm", "M"), ("jr", "R")):
        j = np.asarray(model["affine_" + key])
        z = z + np.asarray(blocks[block]) @ j.reshape(len(j), -1).T
    base = centered_logits(z)
    result["affine_base"] = base
    result["affine_base_score"] = scores(base, model["values"])
    for arm in ARMS:
        inc = result[arm + "_singleton"]
        local = (
            np.stack([inc[:, i] + inc[:, j] for i, j in PAIRS], 1)
            + result[arm + "_mixed"]
        )
        result[arm + "_affine_full"] = centered_logits(base[:, None] + local)
        result[arm + "_affine_score"] = scores(
            result[arm + "_affine_full"], model["values"]
        )
    return result
