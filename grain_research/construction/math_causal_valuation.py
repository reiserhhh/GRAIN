#!/usr/bin/env python3
"""Cause-qualified behavior effects, keeping context choice as evidence."""

from __future__ import annotations

import numpy as np


def cause_parts(fields):
    """Split actual recipient, K donor, V donor and full donor output fields."""
    x = np.asarray(fields, dtype=np.float64)
    if x.shape[0] != 4 or not np.isfinite(x).all():
        raise ValueError("Complete recipient/K/V/KV fields required")
    return np.stack((x[1] - x[0], x[2] - x[0], x[3] - x[1] - x[2] + x[0]))


def behavior_effects(corners):
    """Return assigned/chosen action slopes, cause level and interaction."""
    x = np.asarray(corners, dtype=np.float64)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete b/c corners required")
    slopes = x[1] - x[0]
    level = (x[:, 1] - x[:, 0]).mean(axis=0)
    return slopes, level, slopes[1] - slopes[0]


def install_cause(base_k, base_v, donor_k, donor_v, indices, mode):
    """Replace only explicitly registered cause rows of historical K/V."""
    k, v, dk, dv = (np.asarray(a) for a in (base_k, base_v, donor_k, donor_v))
    slots = np.asarray(indices, dtype=int)
    if k.shape != dk.shape or v.shape != dv.shape or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Registered layer/head/history/coordinate arrays required")
    if mode not in ("k", "v", "kv", "noop"):
        raise ValueError("Unknown cause interchange")
    if (
        slots.ndim != 1
        or not len(slots)
        or len(np.unique(slots)) != len(slots)
        or slots.min() < 0
        or slots.max() >= k.shape[2]
        or slots.max() >= v.shape[2]
        or not all(np.isfinite(a).all() for a in (k, v, dk, dv))
    ):
        raise ValueError("Finite distinct addressed cause rows required")
    out_k, out_v = k.copy(), v.copy()
    if mode in ("k", "kv", "noop"):
        out_k[:, :, slots] = dk[:, :, slots]
    if mode in ("v", "kv", "noop"):
        out_v[:, :, slots] = dv[:, :, slots]
    return out_k, out_v


def posterior_odds(prior, action_likelihoods, context_likelihoods=(1.0, 1.0)):
    """Two-theta Bayes calculation; context selection is not removed as noise."""
    a, z = np.asarray(action_likelihoods), np.asarray(context_likelihoods)
    if not 0 < prior < 1 or a.shape != (2,) or z.shape != (2,):
        raise ValueError("A proper prior and two likelihood pairs required")
    if not all(
        np.isfinite(x).all() and (x > 0).all() and (x <= 1).all() for x in (a, z)
    ):
        raise ValueError("Strictly positive finite probability likelihoods required")
    return float(prior / (1 - prior) * a[1] / a[0] * z[1] / z[0])
