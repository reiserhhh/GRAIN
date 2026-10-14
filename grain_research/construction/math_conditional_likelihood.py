#!/usr/bin/env python3
"""Conditional predictive evidence accounts, not an LLM personality judgement."""

from __future__ import annotations

import numpy as np


def paired_account(chosen, partition, direct_logprob, logit_mean):
    """Keep selected-token action and alternative competition in one unit."""
    arrays = [
        np.asarray(x, float) for x in (chosen, partition, direct_logprob, logit_mean)
    ]
    if any(x.ndim != 2 or x.shape[0] != 2 for x in arrays):
        raise ValueError("Two hypotheses by aligned observed tokens required")
    if any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Finite aligned accounts required")
    z, logz, lp, center = arrays
    numerator = z[0] - z[1]
    normalizer = logz[0] - logz[1]
    ratio = numerator - normalizer
    delta_center = center[0] - center[1]
    return {
        "numerator": numerator,
        "normalizer": normalizer,
        "ratio": ratio,
        "centered_numerator": numerator - delta_center,
        "centered_normalizer": normalizer - delta_center,
        "direct_ratio": lp[0] - lp[1],
        "account_error": float(np.max(np.abs(ratio - (lp[0] - lp[1])))),
    }


def slice_account(carry, reset, boundary=16):
    """Separate inherited-prefix evidence from a reset observation model."""
    x = np.asarray(carry, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or boundary < 1:
        raise ValueError("Finite nonempty path and positive boundary required")
    first = float(x[:boundary].sum())
    full = float(x.sum())
    result = {"first_slice": first, "full": full, "tokens": len(x)}
    if len(x) <= boundary:
        if reset is not None:
            raise ValueError("No second slice exists")
        return {
            **result,
            "carry_second": None,
            "reset_second": None,
            "reset_sum": None,
            "carry_minus_reset": None,
        }
    y = np.asarray(reset, float)
    if y.shape != (len(x) - boundary,) or not np.isfinite(y).all():
        raise ValueError("The same second-slice tokens must be scored")
    second = float(x[boundary:].sum())
    reset_second = float(y.sum())
    return {
        **result,
        "carry_second": second,
        "reset_second": reset_second,
        "reset_sum": first + reset_second,
        "carry_minus_reset": second - reset_second,
        "chain_error": abs(full - first - second),
    }
