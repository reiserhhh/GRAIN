#!/usr/bin/env python3
"""Query-independent local GRAIN/embedding operator for compiled linear queries."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.material_geometry import cross_covariance


def build_operator(grain, embeddings):
    """Build C_gh from aligned comments; no target or query direction is needed."""
    return cross_covariance(grain, embeddings)


def read_operator(operator, directions):
    """Read one or many fixed directions as exact local cross-covariances."""
    c, w = np.asarray(operator, float), np.asarray(directions, float)
    if c.ndim != 2 or w.ndim not in (1, 2) or c.shape[1] != w.shape[0]:
        raise ValueError("Operator and query coordinates must align")
    if not np.isfinite(c).all() or not np.isfinite(w).all():
        raise ValueError("Finite operator and directions required")
    return c @ w
