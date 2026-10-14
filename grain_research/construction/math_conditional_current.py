#!/usr/bin/env python3
"""Condition-indexed currents and source-common opportunity-preserving nulls."""

from __future__ import annotations

import itertools

import numpy as np


def cycles(mass):
    """Project signed pair mass on complete-graph cycles, vectorized over cells."""
    x = np.asarray(mass, float)
    if x.ndim < 2 or x.shape[-1] != x.shape[-2] or x.shape[-1] < 2:
        raise ValueError("Finite square state axes required")
    if not np.isfinite(x).all():
        raise ValueError("Finite mass required")
    a = (x - x.swapaxes(-1, -2)) / 2
    potential = a.sum(-1) / x.shape[-1]
    return a - (potential[..., :, None] - potential[..., None, :])


def lift(counts, references):
    """Retain cells and a non-destructive visible/hidden direct-sum account."""
    c, refs = np.asarray(counts, float), np.asarray(references, float)
    if c.ndim != 6 or refs.ndim != 5 or c.shape[2:] != refs.shape[1:]:
        raise ValueError(
            "Author/view/context/position/state counts and arm references required"
        )
    if c.min() < 0 or not np.isfinite(c).all() or not np.isfinite(refs).all():
        raise ValueError("Nonnegative finite counts required")
    if refs.min() < 0 or not np.allclose(refs.sum((-2, -1)), 1):
        raise ValueError("Probability references required")
    total = c.sum((2, 3, 4, 5))
    if (total <= 0).any():
        raise ValueError("Nonempty author views required")
    raw = c / total[..., None, None, None, None]
    q = raw.sum((-2, -1))
    raw = np.broadcast_to(raw[:, :, None], (*c.shape[:2], len(refs), *c.shape[2:]))
    opportunity = q[:, :, None, :, :, None, None] * refs[None, None]
    laws = np.stack((raw, opportunity, raw - opportunity), axis=3)
    cell = cycles(laws)
    cells, ns = c.shape[2] * c.shape[3], c.shape[-1]
    cell = cell.reshape(*cell.shape[:4], cells, ns, ns)
    flat = cell.sum(-3)
    hidden = cell - flat[..., None, :, :] / cells
    return {
        "resolved": cell.reshape(*cell.shape[:4], -1),
        "visible": (flat / np.sqrt(cells)).reshape(*flat.shape[:4], -1),
        "hidden": hidden.reshape(*hidden.shape[:4], -1),
    }, q


def author_covariance(feature, strata):
    """Return whole-author contributions to centered cross-view covariance."""
    x, groups = np.asarray(feature, float), np.asarray(strata)
    if x.ndim < 3 or x.shape[1] != 3 or groups.shape != (len(x),):
        raise ValueError("Three aligned views and strata required")
    if not np.isfinite(x).all():
        raise ValueError("Finite features required")
    result = np.empty(x.shape[:1] + x.shape[2:-1])
    for group in np.unique(groups):
        selected = groups == group
        n = int(selected.sum())
        if n < 2:
            raise ValueError("At least two authors per stratum required")
        z = x[selected] - x[selected].mean(0)
        terms = [(z[:, a] * z[:, b]).sum(-1) for a, b in ((0, 1), (0, 2), (1, 2))]
        result[selected] = np.mean(terms, axis=0) * n / (n - 1)
    return result


def source_markov(source_counts, alpha):
    """Estimate one shared position-inhomogeneous four-slice null from source only."""
    counts = np.asarray(source_counts, float)
    if counts.ndim != 5 or counts.shape[2] != 3 or counts.shape[-1] != counts.shape[-2]:
        raise ValueError("Source/context/three-position/state counts required")
    if counts.min() < 0 or alpha <= 0 or not np.isfinite(counts).all():
        raise ValueError("Nonnegative source counts and positive pseudocount required")
    total, ns = counts.sum(0), counts.shape[-1]
    initial = total[:, 0].sum(-1) + alpha
    initial /= initial.sum(-1, keepdims=True)
    transitions = total + alpha
    transitions /= transitions.sum(-1, keepdims=True)
    patterns = np.array(list(itertools.product(range(ns), repeat=4)), int)
    probabilities = initial[:, patterns[:, 0]].copy()
    for r in range(3):
        probabilities *= transitions[:, r, patterns[:, r], patterns[:, r + 1]]
    if not np.allclose(probabilities.sum(1), 1):
        raise AssertionError("Shared comment laws failed to normalize")
    return patterns, probabilities, initial, transitions


def counts_for_comments(states, authors, views, conditions, n_authors, n_conditions):
    """Count whole four-state comments without breaking adjacency or adding wrap."""
    x = np.asarray(states, int)
    u, v, z = (np.asarray(a, int) for a in (authors, views, conditions))
    if x.shape != (len(u), 4) or v.shape != u.shape or z.shape != u.shape:
        raise ValueError("Aligned comment addresses required")
    if (
        (x < 0).any()
        or (x >= 4).any()
        or (u < 0).any()
        or (u >= n_authors).any()
        or (v < 0).any()
        or (v >= 3).any()
        or (z < 0).any()
        or (z >= n_conditions).any()
    ):
        raise ValueError("Outside frozen support")
    out = np.zeros((n_authors, 3, n_conditions, 3, 4, 4))
    for r in range(3):
        np.add.at(out, (u, v, z, r, x[:, r], x[:, r + 1]), 1)
    return out


def sample_comments(patterns, probabilities, conditions, rng):
    """Draw independent legal whole comments, keeping condition addresses fixed."""
    z = np.asarray(conditions, int)
    if z.ndim != 1 or z.min() < 0 or z.max() >= len(probabilities):
        raise ValueError("Valid condition address required")
    draws = rng.random(len(z))
    cdf = np.cumsum(probabilities, axis=1)
    cdf[:, -1] = 1.0
    indices = (draws[:, None] >= cdf[z]).sum(1)
    return patterns[indices]


def walsh_patterns():
    """Return three fixed balanced orthogonal contrasts across four contexts."""
    return np.array([[-1, -1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]], float)


def fisher_witness():
    """Compute finite Fisher information before and after forgetting context."""
    base = np.full((4, 3, 3), 1 / 36)
    cycle = np.zeros((3, 3))
    for i in range(3):
        cycle[i, (i + 1) % 3], cycle[(i + 1) % 3, i] = 1 / 15, -1 / 15
    tangents = walsh_patterns()[:, :, None, None] * cycle[None, None] / 4
    full = np.einsum(
        "ac,bc->ab",
        tangents.reshape(3, -1) / np.sqrt(base).ravel(),
        tangents.reshape(3, -1) / np.sqrt(base).ravel(),
    )
    lost_context = tangents.sum(1)
    pooled = np.einsum("aij,bij,ij->ab", lost_context, lost_context, 1 / base.sum(0))
    return base, tangents, full, pooled
