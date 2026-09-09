#!/usr/bin/env python3
"""Signed criterion covariance allocation, without psychological mediation claims."""

from __future__ import annotations

import numpy as np


def criterion_account(total, parts, criterion):
    """Allocate an association using one denominator and retain cancellation."""
    total, parts, y = (np.asarray(a, float) for a in (total, parts, criterion))
    if (
        total.ndim != 1
        or parts.ndim != 2
        or y.shape != total.shape
        or len(parts) != len(total)
    ):
        raise ValueError("Aligned authors, parts and one criterion required")
    if len(total) < 4 or not all(np.isfinite(a).all() for a in (total, parts, y)):
        raise ValueError("At least four finite author observations required")
    if not np.allclose(total, parts.sum(1), atol=1e-10, rtol=1e-10):
        raise ValueError("Components do not reconstruct total")
    yc = y - y.mean()
    tc = total - total.mean()
    pc = parts - parts.mean(0)
    denominator = np.sqrt(np.mean(tc**2) * np.mean(yc**2))
    if denominator <= 1e-14:
        raise ValueError("Nonconstant total and criterion required")
    pieces = (pc.T @ yc) / len(y) / denominator
    r = float(np.mean(tc * yc) / denominator)
    return {
        "total_r": r,
        "contributions": pieces.tolist(),
        "closure_error": float(abs(r - pieces.sum())),
        "denominator": float(denominator),
    }
