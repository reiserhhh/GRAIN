#!/usr/bin/env python3
"""SOURCE-only conditional evidence selection before native reader outputs."""

from itertools import combinations

import numpy as np
from scipy.special import logsumexp


def validate(bank):
    """Require a complete frozen profile alphabet and paired source catalogue."""
    z = np.asarray(bank["source_means"], float)
    g = np.asarray(bank["codes"])
    s2 = float(bank["variance"])
    if (
        z.ndim != 3
        or z.shape[1] != 2
        or g.shape != (len(z),)
        or g.dtype.kind not in "iu"
        or set(g) != set(range(32))
        or not np.isfinite(z).all()
        or not np.isfinite(s2)
        or s2 <= 0
    ):
        raise ValueError("Complete32-profile finite SOURCE catalogue required")
    return z, g, s2


def odds(logs, codes, q):
    """Read signed high/low odds without applying an output fit."""
    high = ((np.asarray(codes) >> q) & 1).astype(bool)
    return logsumexp(logs[:, high], -1) - logsumexp(logs[:, ~high], -1)


def accounts(bank, points):
    """Return full, profile-association and departure scores for allfive targets."""
    z, g, s2 = validate(bank)
    x = np.asarray(points, float)
    if x.ndim != 2 or x.shape[1] != z.shape[2] or not np.isfinite(x).all():
        raise ValueError("Finite single-window point matrix required")
    lp = np.stack(
        [x @ z[:, v].T / s2 - np.sum(z[:, v] ** 2, 1) / (2 * s2) for v in (0, 1)],
        -1,
    )
    lp = logsumexp(lp, -1) - np.log(2)
    counts = np.bincount(g, minlength=32)
    law = np.stack([logsumexp(lp[:, g == c], -1) for c in range(32)], -1)
    law -= np.log(len(g))
    out = np.empty((len(x), 3, 5))
    for q in range(5):
        c = g & ~(1 << q)
        base = np.stack(
            [
                logsumexp(lp[:, c == (code & ~(1 << q))], -1)
                - np.log(np.count_nonzero(c == (code & ~(1 << q))))
                + np.log(counts[code] / len(g))
                for code in range(32)
            ],
            -1,
        )
        f, b = odds(law, np.arange(32), q), odds(base, np.arange(32), q)
        out[:, :, q] = np.column_stack((f, b, f - b))
    return out


def select(bank, vectors, conditions, lengths, *, absolute=8, relative=0.05):
    """Choose a matched disjoint removal pair using SOURCE A departure only."""
    x = np.asarray(vectors, float)
    c, lengths = np.asarray(conditions), np.asarray(lengths)
    if (
        x.ndim != 2
        or x.shape[0] != 12
        or c.shape != (12,)
        or lengths.shape != (12,)
        or not np.isin(c, [0, 1]).all()
        or lengths.dtype.kind not in "iu"
        or np.any(lengths <= 0)
        or not np.isfinite(x).all()
        or absolute < 0
        or not 0 <= relative < 1
    ):
        raise ValueError(
            "Twelve finite addressed events and fixed matching rules required"
        )
    subsets = np.array(list(combinations(range(12), 4)), int)
    points = (x.sum(0) - x[subsets].sum(1)) / 8
    values = accounts(bank, points)
    nc, nt = c[subsets].sum(1), lengths[subsets].sum(1)
    bits = np.sum(1 << subsets, 1)
    feasible = (bits[:, None] & bits[None, :]) == 0
    feasible &= nc[:, None] == nc[None, :]
    feasible &= abs(nt[:, None] - nt[None, :]) <= np.maximum(
        absolute, relative * np.maximum(nt[:, None], nt[None, :])
    )
    feasible &= np.triu(np.ones(feasible.shape, bool), 1)
    if not feasible.any():
        raise ValueError("No supported matched pair; no outcome-driven relaxation")
    d = values[:, 2, 2]
    gaps = np.where(feasible, abs(d[:, None] - d[None, :]), -np.inf)
    i, j = np.unravel_index(np.argmax(gaps), gaps.shape)
    support, counter = (i, j) if d[i] <= d[j] else (j, i)
    return {
        "subsets": subsets,
        "points": points,
        "accounts": values,
        "support": int(support),
        "counter": int(counter),
        "feasible_pairs": int(feasible.sum()),
        "lengths": nt,
        "condition_counts": nc,
    }


def native_accounts(logits):
    """Read fixed digit endpoints; keep generic and target-specific actions."""
    a = np.asarray(logits, float)
    if a.ndim != 4 or a.shape[1:] != (3, 5, 10) or not np.isfinite(a).all():
        raise ValueError("Author x3arms x5targets x10digit logits required")
    y = logsumexp(a[..., 5:], -1) - logsumexp(a[..., :5], -1)
    raw = y[:, 2] - y[:, 1]
    return y, raw, raw - raw.mean(1, keepdims=True)


def bootstrap(values, draws, seed):
    """Resample whole authors, not questions or same-person text fragments."""
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() or draws < 1:
        raise ValueError("Finite author actions required")
    ix = np.random.default_rng(seed).integers(len(x), size=(draws, len(x)))
    return np.quantile(x[ix].mean(1), [0.025, 0.975], axis=0).T
