#!/usr/bin/env python3
"""Lossless admission adapter for the sealed HM123 history-layer guard."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_stage_mass_action as T

SEALED_CONSTRUCT = T.construct


def construct(k, v, early, late, positions, origin, seeds, cos, sin, weights, cfg, cut):
    """Admit one history cache per completed block; the padding is never read."""
    import torch

    blocks = len(weights) - 1
    if k.ndim != 5 or k.shape != v.shape or k.shape[1] != blocks:
        raise ValueError("One history layer per completed block is required")
    unused = torch.zeros_like(k[:, :1])
    return SEALED_CONSTRUCT(
        torch.cat((k, unused), dim=1),
        torch.cat((v, unused), dim=1),
        early,
        late,
        positions,
        origin,
        seeds,
        cos,
        sin,
        weights,
        cfg,
        cut,
    )


def remaining_budget(state, completed_files, original_limit):
    """Debit the pre-recurrence failure, never restart completed calculations."""
    seconds = float(state["seconds"])
    if (
        state["state"] != "FAILED"
        or state["completed_units"] != 0
        or state["attempts"] != 1
        or state["pending"] != [0, 2]
        or completed_files
        or not np.isfinite(seconds)
        or not 0 <= seconds < original_limit
    ):
        raise ValueError("Only the zero-output input-guard failure is admitted")
    return original_limit - seconds
