#!/usr/bin/env python3
"""Role-space coordinate transport for frozen typed action kernels.

This is a change of basis in one reader, not a claim that two neural readers
are related by an invertible map. The transported metric preserves the old
scoring unit rather than normalizing the new vectors again.
"""

from __future__ import annotations

import numpy as np


def role_basis(dimension: int, seed: int, scaled: bool = False) -> dict:
    """Build an invertible signed-permutation basis, optionally with scaling."""
    rng = np.random.default_rng(seed)
    return {
        "permutation": rng.permutation(dimension),
        "sign": rng.choice((-1.0, 1.0), dimension),
        "scale": np.exp(rng.uniform(-np.log(2), np.log(2), dimension))
        if scaled
        else np.ones(dimension),
    }


def transform_role(vectors, basis):
    """Change row-vector coordinates without discarding or recentering data."""
    return np.asarray(vectors)[:, basis["permutation"]] * basis["sign"] * basis["scale"]


def metric_kernel(a, b, ar, br, action_metric, object_metric):
    """Contract the typed tensor kernel with diagonal transported metrics."""
    ka = (np.asarray(a) * action_metric) @ np.asarray(ar).T
    kb = (np.asarray(b) * object_metric) @ np.asarray(br).T
    return (1 + ka) * (1 + kb)


def frozen_kernel_score(model, cross_gram):
    """Apply a frozen centered dual program to a supplied cross-Gram matrix."""
    k = np.asarray(cross_gram)
    center = (
        k
        - k.mean(axis=1, keepdims=True)
        - model["column_mean"][None, :]
        + model["grand_mean"]
    )
    return center @ model["dual"] + model["bias"]


def hadamard_interaction_gram(a, b):
    """Expose the coordinate identification imposed by pointwise multiplication."""
    h = np.asarray(a) * np.asarray(b)
    return h @ h.T
