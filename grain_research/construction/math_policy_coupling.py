#!/usr/bin/env python3
"""Qualified evidence-policy contractions; no model fitting or trait inference."""

from __future__ import annotations

import itertools

import numpy as np

ATOMS = np.array(list(itertools.product(range(6), (-1, 1), (-1, 1))))
POLICIES = ("focus_A", "focus_B", "balanced", "positive_coupled", "negative_coupled")


def recipes():
    """Return all fixed positive behavior/choice laws, without observed outcomes."""
    alternating = (-1.0) ** np.arange(6)
    components = (
        (np.zeros(6), np.zeros(6), np.ones(6) / 6, 0.5),
        (np.full(6, 0.4), np.zeros(6), np.ones(6) / 6, 0.5),
        (np.full(6, -0.4), np.zeros(6), np.ones(6) / 6, 0.5),
        (np.zeros(6), 0.4 * alternating, np.ones(6) / 6, 0.5),
        (0.2 * alternating, 0.3 * alternating, np.ones(6) / 6, 0.5),
        (0.2 * alternating, 0.3 * alternating, np.arange(1, 7) / 21, 0.7),
    )
    laws = []
    for a, slope, family, positive_condition in components:
        law = np.array(
            [
                family[f]
                * (positive_condition if c == 1 else 1 - positive_condition)
                * (1 + b * (a[f] + c * slope[f]))
                / 2
                for f, c, b in ATOMS
            ]
        )
        if np.min(law) <= 0 or not np.isclose(law.sum(), 1):
            raise ValueError("Fixed strictly positive evidence law required")
        laws.append(law)
    return np.stack(laws)


def route_counts(counts, divisor=4):
    """Construct exact equal-margin routing pairs using evidence counts only."""
    n = np.asarray(counts)
    if (
        n.shape != (24,)
        or n.dtype.kind not in "iu"
        or np.any(n < 0)
        or np.any(n % 2)
        or n.sum() <= 0
        or not isinstance(divisor, int)
        or divisor < 2
    ):
        raise ValueError("24 nonnegative even event counts and legal divisor required")
    out = np.zeros((5, 24, 2), dtype=np.int64)
    out[0, :, 1], out[1, :, 0] = n, n
    out[2:] = n[None, :, None] // 2
    for low in range(0, 24, 2):
        high = low + 1
        k = min(int(n[low]), int(n[high])) // divisor
        out[3, high] += (-k, k)
        out[3, low] += (k, -k)
        out[4, high] += (k, -k)
        out[4, low] += (-k, k)
    return out


def score(kernels, joint_counts, evidence_owner, query_owner, origins):
    """Keep observed, reference and provenance-specific policy contributions.

    Kernels are owner/object/evidence/focus/query/unit; ledgers are
    history/policy/evidence/focus. Every admitted endpoint is own-actor evidence.
    Reference scores are a declared instrument expectation, not corrected truth.
    """
    k = np.asarray(kernels, float)
    n = np.asarray(joint_counts)
    owners, queried = np.asarray(evidence_owner), np.asarray(query_owner)
    origins = np.asarray(origins)
    histories = len(n) if n.ndim else 0
    if (
        k.shape != (2, 5, 24, 2, 6, 4)
        or not np.isfinite(k).all()
        or n.shape != (histories, 5, 24, 2)
        or histories == 0
        or n.dtype.kind not in "iu"
        or np.any(n < 0)
        or owners.shape != (histories,)
        or queried.shape != (histories,)
        or not np.isin(owners, (0, 1)).all()
        or not np.array_equal(owners, queried)
        or origins.shape != (histories,)
        or not np.isin(origins, ("instrument", "author")).all()
    ):
        raise ValueError(
            "Complete finite own-actor kernels and typed joint logs required"
        )
    totals = n.sum(axis=(2, 3))
    if np.any(totals <= 0):
        raise ValueError("Unobserved author cannot receive fabricated zero scores")
    weights = n / totals[:, :, None, None]
    selected = k[owners]
    observed = np.einsum("hper,haerqu->hpaqu", weights, selected)
    marginal = weights.sum(axis=3)
    reference = np.einsum("hpe,haequ->hpaqu", marginal, selected.mean(axis=3))
    policy = np.einsum(
        "hper,haerqu->hpaqu", weights - marginal[..., None] / 2, selected
    )
    instrument = policy * (origins == "instrument")[:, None, None, None, None]
    author_choice = policy * (origins == "author")[:, None, None, None, None]
    group_weights = weights.reshape(histories, 5, 12, 2, 2)
    group_counts = n.reshape(histories, 5, 12, 2, 2).sum(axis=(3, 4))
    group_totals = np.einsum(
        "hpgbr,hagbrqu->hpgaqu",
        group_weights,
        selected.reshape(histories, 5, 12, 2, 2, 6, 4),
    )
    means = np.full_like(group_totals, np.nan)
    group_mass = group_weights.sum(axis=(3, 4))
    np.divide(
        group_totals,
        group_mass[..., None, None, None],
        out=means,
        where=group_mass[..., None, None, None] > 0,
    )
    return {
        "observed": observed,
        "reference": reference,
        "policy": policy,
        "instrument": instrument,
        "author_choice": author_choice,
        "counts": totals,
        "evidence_counts": n.sum(axis=3),
        "focus_counts": n.sum(axis=2),
        "group_counts": group_counts,
        "conditional_mean": means,
    }


def direct_pair(kernels, counts, divisor=4):
    """Independently sum the four-corner coupled-policy contrast, all objects."""
    k, n = np.asarray(kernels, float), np.asarray(counts)
    if k.shape != (5, 24, 2, 6, 4) or n.shape != (24,) or n.sum() <= 0:
        raise ValueError(
            "Complete owner-specific kernels and positive event count required"
        )
    result = np.zeros((5, 6, 4))
    for low in range(0, 24, 2):
        high = low + 1
        step = min(int(n[low]), int(n[high])) // divisor
        result += (
            2
            * step
            / n.sum()
            * (k[:, high, 1] - k[:, high, 0] - k[:, low, 1] + k[:, low, 0])
        )
    return result
