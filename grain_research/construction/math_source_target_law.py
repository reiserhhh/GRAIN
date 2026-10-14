#!/usr/bin/env python3
"""Frozen SOURCE-supported joint target laws, not calibrated human posteriors."""

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_prospective_evidence as P


def validate(bank):
    """Require finite paired prototypes and joint SOURCE-only anchor rows."""
    if set(bank) != {"source_means", "codes", "variance", "anchors"}:
        raise ValueError("Only SOURCE prototypes, codes, variance and anchors allowed")
    z, codes, variance = P.validate(bank)
    y = np.asarray(bank["anchors"], float)
    if y.shape != (len(z), 5) or not np.isfinite(y).all():
        raise ValueError("Five finite SOURCE anchor columns required")
    return z, codes, variance, y


def weights(bank, points):
    """Normalize inherited two-window Gaussian similarity over SOURCE authors."""
    z, _, variance, _ = validate(bank)
    x = np.asarray(points, float)
    if x.ndim != 2 or x.shape[1] != z.shape[2] or not np.isfinite(x).all():
        raise ValueError("Finite point matrix with frozen embedding width required")
    lp = np.stack(
        [
            x @ z[:, v].T / variance - (z[:, v] ** 2).sum(1) / (2 * variance)
            for v in range(2)
        ],
        -1,
    )
    lp = logsumexp(lp, -1) - np.log(2)
    return np.exp(lp - logsumexp(lp, -1, keepdims=True))


def validate_laws(mass, anchors):
    """Validate distributions without renormalizing invalid caller inputs."""
    w, y = np.asarray(mass, float), np.asarray(anchors, float)
    if (
        w.ndim != 2
        or y.ndim != 2
        or y.shape != (w.shape[1], 5)
        or not len(w)
        or not len(y)
        or not np.isfinite(w).all()
        or not np.isfinite(y).all()
        or np.any(w < 0)
        or not np.allclose(w.sum(1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Finite normalized joint SOURCE masses required")
    return w, y


def cdf(mass, anchors, target, thresholds):
    """Evaluate a marginal CDF with all ties included at each threshold."""
    w, y = validate_laws(mass, anchors)
    t = np.asarray(thresholds, float)
    if target not in range(5) or t.ndim != 1 or not np.isfinite(t).all():
        raise ValueError("Valid target and finite thresholds required")
    return w @ (y[:, target, None] <= t).astype(float)


def superiority(mass, anchors, other=None):
    """Compute independent-draw P(Y_left > Y_right) plus half tie mass."""
    w, y = validate_laws(mass, anchors)
    v, _ = validate_laws(w if other is None else other, y)
    out = np.empty((len(w), len(v), 5))
    for q in range(5):
        a, b = y[:, q, None], y[None, :, q]
        kernel = (a > b).astype(float) + 0.5 * (a == b)
        out[:, :, q] = w @ kernel @ v.T
    return out


def summaries(bank, mass):
    """Derive levels, joint covariance and reference comparisons from one law."""
    _, codes, _, anchors = validate(bank)
    w, y = validate_laws(mass, anchors)
    mean = w @ y
    centered = y[None] - mean[:, None]
    covariance = np.einsum("ui,uiq,uir->uqr", w, centered, centered)
    high = w @ ((codes[:, None] >> np.arange(5)) & 1)
    if np.any((high <= 0) | (high >= 1)):
        raise ValueError("Finite high/low projections lost numerical support")
    reference = np.full((1, len(y)), 1 / len(y))
    return {
        "mean": mean,
        "reference_delta": mean - y.mean(0),
        "covariance": covariance,
        "variance": np.diagonal(covariance, axis1=1, axis2=2),
        "high_probability": high,
        "high_log_odds": np.log(high) - np.log1p(-high),
        "reference_superiority": superiority(w, y, reference)[:, 0],
        "effective_source_authors": 1 / np.sum(w**2, 1),
        "source_mass_tv_from_uniform": 0.5 * abs(w - reference).sum(1),
    }


def points(vectors, event_ids, support, counter):
    """Keep the inherited complete history and two addressed removal arms."""
    x = np.asarray(vectors, float)
    ids = list(event_ids)
    s, c = np.asarray(support), np.asarray(counter)
    if (
        x.ndim != 2
        or len(x) != 12
        or not np.isfinite(x).all()
        or len(ids) != 12
        or len(set(ids)) != 12
        or any(not isinstance(e, str) or not e for e in ids)
        or any(a.shape != (4,) or a.dtype.kind not in "iu" for a in (s, c))
        or any(len(set(a.tolist())) != 4 or np.any((a < 0) | (a >= 12)) for a in (s, c))
        or set(s.tolist()) & set(c.tolist())
    ):
        raise ValueError(
            "Twelve unique addressed events and disjoint removals required"
        )
    return np.stack(
        (x.mean(0), (x.sum(0) - x[s].sum(0)) / 8, (x.sum(0) - x[c].sum(0)) / 8)
    )


def score(bank, entry):
    """Cold-score numeric evidence; reject application labels or model answers."""
    if set(entry) != {"author", "event_ids", "vectors", "support", "counter"}:
        raise ValueError("Unexpected application inputs; no labels or answers allowed")
    if not isinstance(entry["author"], int) or isinstance(entry["author"], bool):
        raise TypeError("An integer opaque author address is required")
    x = points(entry["vectors"], entry["event_ids"], entry["support"], entry["counter"])
    w = weights(bank, x)
    return {"author": entry["author"], "weights": w, **summaries(bank, w)}


def directed_cycle(edges, differences, n, margin=1e-12):
    """Return one strict directed cycle, or no witness on the supplied graph."""
    e, d = np.asarray(edges), np.asarray(differences, float)
    if (
        e.ndim != 2
        or e.shape[1] != 2
        or e.dtype.kind not in "iu"
        or d.shape != (len(e),)
        or not np.isfinite(d).all()
        or np.any((e < 0) | (e >= n))
        or margin < 0
    ):
        raise ValueError("Finite addressed pair graph required")
    adjacency = [[] for _ in range(n)]
    for (i, j), v in zip(e, d, strict=True):
        if v > margin:
            adjacency[i].append(int(j))
        elif v < -margin:
            adjacency[j].append(int(i))
    state, path = np.zeros(n, int), []

    def visit(u):
        state[u] = 1
        path.append(u)
        for v in adjacency[u]:
            if state[v] == 1:
                return path[path.index(v) :] + [v]
            if state[v] == 0:
                found = visit(v)
                if found:
                    return found
        path.pop()
        state[u] = 2
        return []

    for u in range(n):
        if state[u] == 0:
            found = visit(u)
            if found:
                return found
    return []
