#!/usr/bin/env python3
"""Source-anchored joint slice witnesses, independent of LLM answer imitation."""

from __future__ import annotations

import itertools

import numpy as np


def checked_views(values):
    """Validate a nonempty finite four-view/four-coordinate observation array."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (4, 4) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Nonempty finite n by4 by4 views required")
    return x


def source_units(values):
    """Construct per-view origins and rotation-invariant scalar source units."""
    x = checked_views(values)
    origin = x.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((x - origin) ** 2, axis=2), axis=0))
    if np.any(scale <= 0):
        raise ValueError("A source view has no observed variation")
    return origin, scale


def standardize(values, origin, scale):
    """Apply frozen source units without re-centering the application cohort."""
    x = checked_views(values)
    m, s = np.asarray(origin, float), np.asarray(scale, float)
    if (
        m.shape != (4, 4)
        or s.shape != (4,)
        or not np.isfinite(m).all()
        or not np.isfinite(s).all()
        or np.any(s <= 0)
    ):
        raise ValueError("Finite4 by4 origin and four positive scales required")
    return (x - m) / s[None, :, None]


def feature_sectors():
    """Assign each of625 tensor coordinates to its exact nonconstant view set."""
    return np.asarray(
        [
            sum(1 << v for v, index in enumerate(indices) if index != 0)
            for indices in itertools.product(range(5), repeat=4)
        ],
        dtype=int,
    )


def tensor_features(values):
    """Return the explicit tensor product of four affine five-coordinate views."""
    z = checked_views(values)
    affine = np.concatenate((np.ones((len(z), 4, 1)), z), axis=2)
    result = affine[:, 0]
    for v in range(1, 4):
        result = np.einsum("ni,nj->nij", result, affine[:, v]).reshape(len(z), -1)
    return result


def product_kernel(left, right):
    """Compute the equivalent kernel without forming the625-dimensional tensor."""
    x, y = checked_views(left), checked_views(right)
    result = np.ones((len(x), len(y)))
    for v in range(4):
        result *= 1 + x[:, v] @ y[:, v].T
    return result


def anchor_weights(axes, poles, targets):
    """Use only source intervention assignments to orient empirical witnesses."""
    axes, poles = np.asarray(axes), np.asarray(poles)
    if (
        axes.ndim != 1
        or poles.shape != axes.shape
        or not np.isin(poles, (-1, 0, 1)).all()
    ):
        raise ValueError("Aligned source assignment metadata required")
    weights = np.zeros((len(axes), len(targets)))
    for k, name in enumerate(targets):
        for pole in (-1, 1):
            selected = (axes == name) & (poles == pole)
            if not selected.any():
                raise ValueError("Both observed source assignment poles required")
            weights[selected, k] = pole / selected.sum()
    return weights


def relink_source(values, groups, seeds):
    """Change seed pairing while retaining every conditional per-view multiset."""
    x = checked_views(values)
    if len(groups) != len(x) or len(seeds) != len(x):
        raise ValueError("Aligned source groups and seeds required")
    result = x.copy()
    for group in sorted(set(groups)):
        ids = [i for i, g in enumerate(groups) if g == group]
        ids.sort(key=lambda i: seeds[i])
        if len(ids) != 3 or len({seeds[i] for i in ids}) != 3:
            raise ValueError("Three distinct seeds required in every source cell")
        for v, shift in enumerate((0, 1, 2, 1)):
            result[ids, v] = x[np.roll(ids, shift), v]
    return result


def compile_witness(source, weights, relinked):
    """Compile all-sector and relinked functions in one unchanged source unit."""
    x, r = checked_views(source), checked_views(relinked)
    a = np.asarray(weights, float)
    if (
        r.shape != x.shape
        or a.ndim != 2
        or a.shape[0] != len(x)
        or not np.isfinite(a).all()
        or not np.allclose(a.sum(axis=0), 0, atol=1e-12, rtol=0)
    ):
        raise ValueError("Aligned source views and zero-mass source contrasts required")
    full = a.T @ tensor_features(x)
    shuffled = a.T @ tensor_features(r)
    norms = np.linalg.norm(full, axis=1)
    if np.any(norms <= 0):
        raise ValueError("A source contrast has no measured witness")
    sectors = feature_sectors()
    coefficients = np.stack(
        (full, full * (np.asarray([m.bit_count() <= 1 for m in sectors])), shuffled),
        axis=0,
    )
    return coefficients / norms[None, :, None], norms


def score_views(values, origin, scale, coefficients):
    """Cold numeric scoring; inputs contain no labels, IDs, tasks or answers."""
    features = tensor_features(standardize(values, origin, scale))
    w = np.asarray(coefficients, float)
    if w.ndim != 3 or w.shape[0] != 3 or w.shape[2] != 625 or not np.isfinite(w).all():
        raise ValueError(
            "Three finite arms of compiled625-coordinate witnesses required"
        )
    sectors = feature_sectors()
    parts = np.stack(
        [
            features[:, sectors == mask]
            @ w[:, :, sectors == mask]
            .transpose(2, 0, 1)
            .reshape((sum(sectors == mask), -1))
            for mask in range(16)
        ],
        axis=-1,
    )
    parts = parts.reshape(len(features), 3, w.shape[1], 16)
    return parts.sum(axis=-1), parts
