#!/usr/bin/env python3
"""Allocate frozen-function remainders to already measured conditional terms."""

from __future__ import annotations

import numpy as np

from .criterion_accounting import criterion_account


def constant_fiber_loss(target, constant):
    """Exact minimum constant risk on one finite, equally weighted input fiber."""
    t = np.asarray(target, float)
    if (
        t.ndim != 1
        or len(t) < 2
        or not np.isfinite(t).all()
        or not np.isfinite(constant)
    ):
        raise ValueError("Finite target family and constant required")
    variance = float(np.var(t))
    bias = float((t.mean() - constant) ** 2)
    mse = float(np.mean((t - constant) ** 2))
    return {
        "n": len(t),
        "minimum_constant_mse": variance,
        "fixed_constant_mse": mse,
        "squared_mean_bias": bias,
        "anchor_sd": float(t.std()),
        "anchor_range": float(np.ptp(t)),
        "identity_error": abs(mse - variance - bias),
    }


def projected_components(total, parts, proxy, criterion):
    """Use one affine projector for every term, retaining their aligned total."""
    t, a, p, y = (np.asarray(v, float) for v in (total, parts, proxy, criterion))
    criterion_account(t, a, y)
    if p.shape != t.shape or not np.isfinite(p).all() or p.std() < 1e-12:
        raise ValueError("Aligned finite nonconstant proxy required")
    pc, ac, tc = p - p.mean(), a - a.mean(0), t - t.mean()
    slopes = pc @ ac / (pc @ pc)
    residuals = ac - pc[:, None] * slopes
    aligned = t.mean() + pc * (pc @ tc / (pc @ pc))
    account = criterion_account(t, np.column_stack([aligned, residuals]), y)
    return {
        **account,
        "orthogonality_error": float(np.max(np.abs(pc @ residuals / len(p)))),
        "max_reconstruction_error": float(
            np.max(np.abs(t - aligned - residuals.sum(1)))
        ),
    }


def projected_component_draws(total, parts, proxy, criterion, indices):
    """Recompute a common-denominator conditional account in author resamples."""
    projected_components(total, parts, proxy, criterion)
    ix = np.asarray(indices)
    if (
        ix.ndim != 2
        or ix.shape[1] != len(total)
        or not ix.size
        or not np.issubdtype(ix.dtype, np.integer)
        or ix.min() < 0
        or ix.max() >= len(total)
    ):
        raise ValueError("Whole-author paired integer resamples required")
    t, a, p, y = (np.asarray(v, float)[ix] for v in (total, parts, proxy, criterion))
    tc, ac, pc, yc = (
        t - t.mean(1, keepdims=True),
        a - a.mean(1, keepdims=True),
        p - p.mean(1, keepdims=True),
        y - y.mean(1, keepdims=True),
    )
    vp = (pc**2).mean(1)
    denom = np.sqrt((tc**2).mean(1) * (yc**2).mean(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = (ac * pc[:, :, None]).mean(1) / vp[:, None]
        remainder_cov = (ac * yc[:, :, None]).mean(1) - slopes * (pc * yc).mean(1)[
            :, None
        ]
        aligned_cov = (pc * tc).mean(1) / vp * (pc * yc).mean(1)
        result = np.column_stack([aligned_cov, remainder_cov]) / denom[:, None]
    return np.where(((vp > 1e-24) & (denom > 1e-14))[:, None], result, np.nan)
