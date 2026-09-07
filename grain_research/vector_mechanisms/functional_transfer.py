#!/usr/bin/env python3
"""Frozen gradient averaging and equal-view functional response scoring."""

from __future__ import annotations

import numpy as np


def shared_gradient_scores(gradients, displacements):
    """Return an uncalibrated shared covector, per-view and equal-view scores."""
    g, d = np.asarray(gradients, float), np.asarray(displacements, float)
    if (
        g.ndim != 2
        or d.ndim != 3
        or min(g.shape) == 0
        or min(d.shape) == 0
        or g.shape[1] != d.shape[2]
        or not np.isfinite(g).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError("Finite training gradients and author/view vectors required")
    reference = g.mean(axis=0)
    views = d @ reference
    return reference, views, views.mean(axis=1)
