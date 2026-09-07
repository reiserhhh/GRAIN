#!/usr/bin/env python3
"""Exact mean-null permutation covariance operators for contextual token states."""

from __future__ import annotations

import numpy as np


def permutation_covariance_action(centered_states, direction):
    """Apply a comment's vectorized permutation covariance without materializing it.

    Rows are token positions. For n>1 the operator is
    n/(n-1) * (I - 11'/n) tensor C, where C=R'R/n.
    The row-matrix formula preserves the same row-major vectorization.
    """
    r, v = np.asarray(centered_states, float), np.asarray(direction, float)
    if r.ndim != 2 or r.shape != v.shape or not np.isfinite([r, v]).all():
        raise ValueError("Require finite paired token-state matrices")
    if not len(r) or not np.allclose(r.mean(0), 0, atol=1e-8):
        raise ValueError("Require nonempty within-comment centered states")
    if len(r) == 1:
        return np.zeros_like(v)
    centered_v = v - v.mean(0, keepdims=True)
    return (centered_v @ r.T) @ r / (len(r) - 1)


def mean_null_project(states):
    """Return within-comment residuals; neither residual nor mean is called noise."""
    h = np.asarray(states, float)
    if h.ndim != 2 or not len(h) or not np.isfinite(h).all():
        raise ValueError("Require a finite nonempty token-state matrix")
    return h - h.mean(0, keepdims=True)
