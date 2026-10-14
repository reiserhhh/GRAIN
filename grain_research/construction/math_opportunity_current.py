#!/usr/bin/env python3
"""Separate opportunity composition from source-relative directed response."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_author_current import hodge_current


def components(joint):
    """Project finite signed pair mass; residuals need not be probability laws."""
    j = np.asarray(joint, float)
    if j.ndim != 2 or j.shape[0] != j.shape[1] or not np.isfinite(j).all():
        raise ValueError("Finite square pair mass required")
    skew = (j - j.T) / 2
    gradient, cycle = hodge_current(skew)
    return np.stack(((j + j.T) / 2, gradient, cycle))


def pair_counts(states, conditions, n_states, n_conditions):
    """Keep within-comment edges, positions and context; never wrap endpoints."""
    x, z = np.asarray(states), np.asarray(conditions)
    if x.ndim != 2 or x.shape[1] != 4 or z.shape != (len(x),) or not len(x):
        raise ValueError("Four ordered states per nonempty comment required")
    if not np.equal(x, np.floor(x)).all() or not np.equal(z, np.floor(z)).all():
        raise ValueError("Integer states and conditions required")
    if x.min() < 0 or x.max() >= n_states or z.min() < 0 or z.max() >= n_conditions:
        raise ValueError("States outside frozen support")
    out = np.zeros((n_conditions, 3, n_states, n_states), float)
    for r in range(3):
        np.add.at(
            out, (z.astype(int), r, x[:, r].astype(int), x[:, r + 1].astype(int)), 1
        )
    return out


def reference_account(counts, reference):
    """Retain raw,opportunity,response pair mass under one frozen reference."""
    c, ref = np.asarray(counts, float), np.asarray(reference, float)
    if c.shape != ref.shape or c.ndim != 4 or c.min() < 0 or c.sum() <= 0:
        raise ValueError("Aligned context/position/state counts required")
    if ref.min() < 0 or not np.allclose(ref.sum((-2, -1)), 1):
        raise ValueError("Reference cells must be probability laws")
    q = c.sum((-2, -1)) / c.sum()
    raw = c.sum((0, 1)) / c.sum()
    opportunity = np.einsum("zr,zrij->ij", q, ref)
    return np.stack((raw, opportunity, raw - opportunity)), q


def build_references(author_counts, alpha, minimum_authors):
    """Fit only source references, retaining explicit sparse-cell fallbacks."""
    a = np.asarray(author_counts, float)
    if a.ndim != 5 or a.min() < 0 or alpha <= 0 or minimum_authors < 1:
        raise ValueError("Nonnegative source counts and positive smoothing required")
    total = a.sum(0)
    support = (a.sum((-2, -1)) > 0).sum(0)
    nz, nr, _, _ = total.shape

    def smooth(x):
        return (x + alpha) / (x.sum((-2, -1), keepdims=True) + alpha * x.shape[-1] ** 2)

    pooled = np.broadcast_to(smooth(total.sum((0, 1))), total.shape).copy()
    position = np.broadcast_to(smooth(total.sum(0))[None], total.shape).copy()
    embedding = np.broadcast_to(smooth(total.sum(1))[:, None], total.shape).copy()
    joint = smooth(total)
    sparse = support < minimum_authors
    joint[sparse] = position[sparse]
    embedding_sparse = (a.sum((2, 3, 4)) > 0).sum(0) < minimum_authors
    embedding[embedding_sparse] = position[embedding_sparse]
    if (nz, nr) != (a.shape[1], 3):
        raise ValueError("Three slice positions required")
    return (
        {
            "pooled": pooled,
            "position": position,
            "embedding": embedding,
            "joint": joint,
        },
        support,
        sparse,
    )


def nearest(values, centers):
    """Apply frozen centroids without updating the application distribution."""
    x, c = np.asarray(values, float), np.asarray(centers, float)
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]:
        raise ValueError("Aligned matrix and centers required")
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("Finite vectors required")
    return np.argmin(np.sum((x[:, None] - c[None]) ** 2, -1), 1)


def matched_author_closeness(features, opportunity, k):
    """Compare true authors to context-matched strangers, without score selection."""
    f, q = np.asarray(features, float), np.asarray(opportunity, float)
    if f.ndim != 3 or q.ndim != 3 or f.shape[:2] != q.shape[:2] or len(f) <= k:
        raise ValueError("Author by three views and enough distinct strangers required")
    rows, matching = [], []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        context_distance = np.sum((q[:, a, None] - q[None, :, b]) ** 2, -1)
        np.fill_diagonal(context_distance, np.inf)
        negatives = np.argsort(context_distance, axis=1, kind="stable")[:, :k]
        positive = np.sum((f[:, a] - f[:, b]) ** 2, -1)
        negative = np.sum((f[:, a, None] - f[negatives, b]) ** 2, -1)
        score = (
            (negative > positive[:, None]) + 0.5 * (negative == positive[:, None])
        ).mean(1)
        rows.append(score)
        matching.append(np.take_along_axis(context_distance, negatives, 1).mean(1))
    return np.stack(rows, 1), np.stack(matching, 1)


def bootstrap_mean(values, draws, seed):
    """Resample whole authors, preserving correlated view-pair observations."""
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("At least eight finite author observations required")
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), (draws, len(x)))].mean(1)
    return [float(x.mean()), *map(float, np.quantile(boot, (0.025, 0.975)))]


def toy_worlds():
    """Build exact condition-driven cycles and a distinct planted response cycle."""
    emissions = np.full((3, 3), 0.1)
    np.fill_diagonal(emissions, 0.8)
    conditional = emissions[:, None, :, None] * emissions[None, :, None, :]
    q = np.zeros((3, 3))
    for i in range(3):
        q[i, (i + 1) % 3], q[i, (i - 1) % 3] = 0.8 / 3, 0.2 / 3
    cycle = np.zeros((3, 3))
    for i in range(3):
        cycle[i, (i + 1) % 3], cycle[(i + 1) % 3, i] = 1, -1
    out = {}
    for name, schedule, delta in (
        ("forward_opportunity_only", q, 0),
        ("reverse_opportunity_only", q.T, 0),
        ("unordered_opportunity_only", np.full((3, 3), 1 / 9), 0),
        ("forward_positive_response", q, 0.005),
        ("forward_negative_response", q, -0.005),
    ):
        response = conditional + delta * cycle
        if response.min() < 0 or not np.allclose(response.sum((-2, -1)), 1):
            raise AssertionError("Planted response is not a law")
        observed = np.einsum("ab,abij->ij", schedule, response)
        expected = np.einsum("ab,abij->ij", schedule, conditional)
        matrices = np.stack((observed, expected, observed - expected))
        out[name] = matrices
    return out
