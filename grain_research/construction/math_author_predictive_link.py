#!/usr/bin/env python3
"""Target-conditioned full-text prediction, not a model's personality answer."""

import numpy as np
from scipy.optimize import linear_sum_assignment

TARGETS = ("E", "N", "A", "C", "O")
VOICES = (
    "The writer has no specified behavioral disposition.",
    "The writer usually enjoys meeting people, initiates conversations, and participates in groups.",
    "The writer usually prefers solitary time, stays reserved with unfamiliar people, and avoids attention.",
    "The writer usually worries, is sensitive to stress, and experiences tension when facing problems.",
    "The writer usually stays calm under stress, feels emotionally steady, and recovers after problems.",
    "The writer usually voluntarily helps others, considers their needs, and seeks workable compromises.",
    "The writer usually ignores others' needs, withholds help, and refuses workable compromises.",
    "The writer usually plans tasks, follows commitments, and organizes activities carefully.",
    "The writer usually neglects plans, fails commitments, and leaves activities unorganized.",
    "The writer usually explores unfamiliar ideas, imagines alternatives, and tries new approaches.",
    "The writer usually avoids unfamiliar ideas, dismisses alternatives, and sticks to familiar approaches.",
)
MODES = ("reset", "own", "foreign", "reverse")


def select_pairs(lengths, conditions, maximum):
    """Select the first complete same-condition cross-window pair without labels."""
    n, c = np.asarray(lengths), np.asarray(conditions)
    if n.ndim != 3 or n.shape[1] != 2 or n.shape != c.shape:
        raise ValueError("Author x two windows x addressed events required")
    pairs = []
    for u in range(len(n)):
        candidates = [
            (i, j)
            for i in range(n.shape[2])
            for j in range(n.shape[2])
            if min(n[u, 0, i], n[u, 1, j]) > 0
            and max(n[u, 0, i], n[u, 1, j]) <= maximum
            and c[u, 0, i] == c[u, 1, j]
        ]
        if not candidates:
            raise ValueError("Missing complete author pair; do not reduce scope")
        pairs.append(candidates[0])
    return np.asarray(pairs, int)


def foreign_matching(vectors, lengths, conditions, slots, absolute, relative):
    """Bind a condition/length matched one-to-one stranger map before outcomes."""
    x, n, c, s = map(np.asarray, (vectors, lengths, conditions, slots))
    if x.ndim != 3 or n.shape != x.shape[:2] or c.shape != n.shape:
        raise ValueError("Aligned full-context vector/event family required")
    if s.shape != (len(x),) or np.any(s < 0) or np.any(s >= x.shape[1]):
        raise ValueError("Exact own context slots required")
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=-1) == 0):
        raise ValueError("Finite nonzero context vectors required")
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    cost = np.full((len(x), len(x)), np.inf)
    event = np.full(cost.shape, -1, int)
    for u in range(len(x)):
        for v in range(len(x)):
            if u == v:
                continue
            allowed = np.flatnonzero(
                (c[v] == c[u, s[u]])
                & (abs(n[v] - n[u, s[u]]) <= max(absolute, relative * n[u, s[u]]))
            )
            if not len(allowed):
                continue
            distance = 1 - x[v, allowed] @ x[u, s[u]]
            k = int(np.argmin(distance))
            cost[u, v], event[u, v] = distance[k], allowed[k]
    rows, donors = linear_sum_assignment(cost)
    if (
        not np.array_equal(rows, np.arange(len(x)))
        or not np.isfinite(cost[rows, donors]).all()
    ):
        raise ValueError("No full stranger derangement; do not relax matching")
    return donors, event[rows, donors], cost[rows, donors]


def accounts(chosen, partition):
    """Retain all four context modes and signed vocabulary-normalized contrasts."""
    z, a = np.asarray(chosen, float), np.asarray(partition, float)
    if z.ndim != 4 or z.shape[1:3] != (4, 11) or z.shape != a.shape:
        raise ValueError("Pair x four modes x eleven voices x aligned tokens")
    if not np.isfinite(z).all() or not np.isfinite(a).all():
        raise ValueError("Finite complete observations required")
    numerator = z[:, :, 1::2] - z[:, :, 2::2]
    normalizer = a[:, :, 1::2] - a[:, :, 2::2]
    ratio = numerator - normalizer
    result = {
        "logprob": z - a,
        "numerator": numerator,
        "normalizer": normalizer,
        "ratio": ratio,
    }
    for name, mode in (("stranger", 2), ("order", 3), ("carry", 0)):
        result[name] = ratio[:, 1] - ratio[:, mode]
        result[name + "_numerator"] = numerator[:, 1] - numerator[:, mode]
        result[name + "_normalizer"] = normalizer[:, 1] - normalizer[:, mode]
        result[name + "_neutral"] = (z - a)[:, 1, 0] - (z - a)[:, mode, 0]
    return result


def contact(fields, effects):
    """Describe the complete centered signed field/action contact, without fit."""
    x, y = np.asarray(fields, float), np.asarray(effects, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Aligned author fields and actions required")
    return (x - x.mean(0)).T @ (y - y.mean(0)) / len(x)
