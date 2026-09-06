#!/usr/bin/env python3
"""Exact content-token coverage and descriptive nuisance-conditioned association."""

import numpy as np


def ngram_coverage(current, context, n):
    """Mark each token participating in a current n-gram present in the context."""
    if n < 1:
        raise ValueError("Positive n required")
    current, context = list(current), list(context)
    grams = {tuple(context[i : i + n]) for i in range(len(context) - n + 1)}
    mask = np.zeros(len(current), dtype=bool)
    for i in range(len(current) - n + 1):
        if tuple(current[i : i + n]) in grams:
            mask[i : i + n] = True
    return mask


def residual_correlation(x, y, design):
    """Return in-sample partial association; never label this causal mediation."""
    x, y, design = np.asarray(x, float), np.asarray(y, float), np.asarray(design, float)
    if (
        len(x) != len(y)
        or len(x) != len(design)
        or not all(np.isfinite(a).all() for a in (x, y, design))
    ):
        raise ValueError("Matched finite arrays required")
    rank = int(np.linalg.matrix_rank(design))
    dof = len(x) - rank
    residuals = (
        np.column_stack((x, y))
        - design @ np.linalg.lstsq(design, np.column_stack((x, y)), rcond=None)[0]
    )
    residuals -= residuals.mean(axis=0)
    norm = np.linalg.norm(residuals, axis=0)
    if dof < 3 or np.min(norm) < 1e-10:
        return np.nan, rank, dof
    return (
        float(np.clip(residuals[:, 0] @ residuals[:, 1] / np.prod(norm), -1, 1)),
        rank,
        dof,
    )


def subset_effect(gap, mask, minimum=4):
    """A whole-case scalar or NaN for insufficient selected tokens."""
    gap, mask = np.asarray(gap, float), np.asarray(mask, bool)
    if gap.shape != mask.shape:
        raise ValueError("Matched tail and mask required")
    return float(gap[mask].mean()) if int(mask.sum()) >= minimum else np.nan
