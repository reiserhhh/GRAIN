#!/usr/bin/env python3
"""Conditional DIF contrast API (F12-03/R28). Separate module: f12_transport.py
is hashed into public bundles and is not modified."""

from __future__ import annotations

import numpy as np


def dif_contrast(score, support, reference, conditions):
    """Per-condition bias of supported scores against a reference and the
    residual condition dependence after the best global affine renorm.

    score, support: (authors, readings); reference: same shape (e.g. expected
    old-unit response); conditions: (readings,). Returns a dict with per-
    condition bias, per-condition residual after renorm, and their ranges.
    Unsupported readings are excluded, never imputed.
    """
    score, support, reference = (np.asarray(x) for x in (score, support, reference))
    z = np.broadcast_to(np.asarray(conditions, float), score.shape)
    if score.shape != support.shape or score.shape != reference.shape:
        raise ValueError("aligned score/support/reference arrays required")
    if support.sum() < 2:
        return {"status": "NO_SUPPORTED_SCORES"}
    s, e, zz = score[support], reference[support], z[support]
    alpha, beta = np.linalg.lstsq(np.column_stack([np.ones_like(s), s]), e, rcond=None)[0]
    resid = alpha + beta * s - e
    levels = np.unique(zz)
    bias = {float(v): float(np.mean(s[zz == v] - e[zz == v])) for v in levels}
    after = {float(v): float(np.mean(resid[zz == v])) for v in levels}
    return {
        "status": "OK",
        "bias": bias,
        "after_renorm": after,
        "dif_range": max(bias.values()) - min(bias.values()),
        "dif_range_after_renorm": max(after.values()) - min(after.values()),
        "renorm": {"offset": float(alpha), "scale": float(beta)},
    }
