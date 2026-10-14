#!/usr/bin/env python3
"""Frozen SOURCE selectors crossed with native reader targets, without fitting."""

from itertools import combinations

import numpy as np
from scipy.special import logsumexp

EVALUATIVE = np.array([1.0, -1.0, 1.0, 1.0, 1.0])


def select_all(accounts, subsets, conditions, lengths, *, absolute=8, relative=0.05):
    """Apply the old matched lexicographic rule to each saved SOURCE departure."""
    a, s = np.asarray(accounts, float), np.asarray(subsets)
    c, length = np.asarray(conditions), np.asarray(lengths)
    expected = np.array(list(combinations(range(12), 4)), int)
    if (
        a.shape != (495, 3, 5)
        or not np.isfinite(a).all()
        or s.dtype.kind not in "iu"
        or not np.array_equal(s, expected)
        or c.shape != (12,)
        or not np.isin(c, [0, 1]).all()
        or length.shape != (12,)
        or length.dtype.kind not in "iu"
        or np.any(length <= 0)
        or absolute < 0
        or not 0 <= relative < 1
    ):
        raise ValueError("Complete saved SOURCE accounts and matched12 events required")
    nc, nt = c[s].sum(1), length[s].sum(1)
    bits = np.sum(1 << s, 1)
    feasible = (bits[:, None] & bits[None, :]) == 0
    feasible &= nc[:, None] == nc[None, :]
    feasible &= abs(nt[:, None] - nt[None, :]) <= np.maximum(
        absolute, relative * np.maximum(nt[:, None], nt[None, :])
    )
    feasible &= np.triu(np.ones(feasible.shape, bool), 1)
    if not feasible.any():
        raise ValueError("No supported matched pair; no relaxation")
    rows = []
    for q in range(5):
        d = a[:, 2, q]
        gap = np.where(feasible, abs(d[:, None] - d[None, :]), -np.inf)
        i, j = np.unravel_index(np.argmax(gap), gap.shape)
        support, counter = (i, j) if d[i] <= d[j] else (j, i)
        rows.append(
            {
                "selector": q,
                "support_index": int(support),
                "counter_index": int(counter),
                "support": s[support].tolist(),
                "counter": s[counter].tolist(),
                "feasible_pairs": int(feasible.sum()),
                "removed_condition_counts": [int(nc[k]) for k in (support, counter)],
                "removed_token_lengths": [int(nt[k]) for k in (support, counter)],
                "source_action": (a[counter] - a[support]).tolist(),
            }
        )
    return rows


def read_actions(logits):
    """Read every author x selector x2 arms x5 targets, keeping signed actions."""
    a = np.asarray(logits, float)
    if a.ndim != 5 or a.shape[1:] != (5, 2, 5, 10) or not np.isfinite(a).all():
        raise ValueError("Complete author x5selector x2arm x5reader x10digit required")
    y = logsumexp(a[..., 5:], -1) - logsumexp(a[..., :5], -1)
    return y, y[:, :, 1] - y[:, :, 0]


def decompose(actions):
    """Test a PREDECLARED evaluation direction, not an answer-fitted axis."""
    x = np.asarray(actions, float)
    if x.ndim != 3 or x.shape[1:] != (5, 5) or not np.isfinite(x).all():
        raise ValueError("Complete finite author x5selector x5reader actions required")
    coefficient = x @ EVALUATIVE / 5
    common = coefficient[..., None] * EVALUATIVE
    remainder = x - common
    diagonal = np.diagonal(remainder, axis1=1, axis2=2)
    # A was observed in HM218; the four other selectors are the new primary.
    primary = diagonal[:, [0, 1, 3, 4]].mean(1)
    interaction = remainder - remainder.mean(1, keepdims=True)
    return {
        "coefficient": coefficient,
        "evaluative": common,
        "remainder": remainder,
        "diagonal": diagonal,
        "primary": primary,
        "interaction": interaction,
    }


def bootstrap(values, draws, seed):
    """Use one fixed whole-author resampling family for all matrix cells."""
    x = np.asarray(values, float)
    if x.ndim < 1 or len(x) < 2 or not np.isfinite(x).all() or draws < 1:
        raise ValueError("At least two finite whole-author observations required")
    ix = np.random.default_rng(seed).integers(len(x), size=(draws, len(x)))
    return np.moveaxis(np.quantile(x[ix].mean(1), [0.025, 0.975], axis=0), 0, -1)


def crossbank_witness(interaction):
    """Keep signed cross-bank contact; do not square noisy matrix estimates."""
    x = np.asarray(interaction, float)
    if x.shape != (24, 5, 5) or not np.isfinite(x).all():
        raise ValueError("Fixed24-author panel and12/12 banks required")
    a, b = x[:12].mean(0), x[12:].mean(0)
    return float(np.sum(a * b) / 25), np.stack((a, b))


def extract_bodies(decoded):
    """Recover old public excerpts; callers must validate exact token replay."""
    start = "The excerpts below were written by one anonymous writer.\n\nExcerpt 1:\n"
    stop = "\nEvidence ends here.\n\nQuestion: "
    if decoded.count(start) != 1 or decoded.count(stop) != 1:
        raise ValueError("Ambiguous old prompt boundaries; no text cleanup")
    body = decoded.split(start, 1)[1].split(stop, 1)[0]
    texts = []
    for k in range(2, 13):
        marker = f"\n\nExcerpt {k}:\n"
        if body.count(marker) != 1:
            raise ValueError("Ambiguous or incomplete excerpt numbering")
        current, body = body.split(marker, 1)
        texts.append(current)
    texts.append(body)
    if any(not t for t in texts):
        raise ValueError("Empty old excerpt; no fallback")
    return texts
