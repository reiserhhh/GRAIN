"""Same-evidence view aggregation without questionnaire-based optimization."""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def interval_token_indices(offsets, intervals):
    """Map disjoint character intervals to whole tokens, even for repeated text."""
    if not intervals or any(not 0 <= a < b for a, b in intervals):
        raise ValueError("Invalid content intervals")
    intervals = sorted(intervals)
    if any(left[1] > right[0] for left, right in pairwise(intervals)):
        raise ValueError("Overlapping content intervals")
    groups = [
        [i for i, (a, b) in enumerate(offsets) if b > a >= lo and b <= hi]
        for lo, hi in intervals
    ]
    if any(not group for group in groups):
        raise ValueError("No complete token inside content interval")
    return [i for group in groups for i in group]


def common_groups(texts, length_functions, *, max_tokens=512, max_records=3):
    """Greedily group full adjacent records within every encoder's token budget."""
    groups, current = [], []
    if not texts or not length_functions or max_records < 1:
        raise ValueError("Empty material or invalid grouping specification")
    for i, text in enumerate(texts):
        if any(f(text) > max_tokens for f in length_functions):
            raise ValueError(
                "Single comment exceeds token budget; no truncation allowed"
            )
        proposal = current + [i]
        joined = "\n\n".join(texts[j] for j in proposal)
        if current and (
            len(proposal) > max_records
            or any(f(joined) > max_tokens for f in length_functions)
        ):
            groups.append(current)
            current = [i]
        else:
            current = proposal
    groups.append(current)
    return groups


def pool_groups(vectors, groups, record_count):
    """Weight joint encodings by record count, preserving equal comment exposure."""
    if sorted(i for g in groups for i in g) != list(range(record_count)):
        raise ValueError("Groups must partition the records exactly once")
    v = np.asarray(vectors)
    if v.ndim != 2 or len(v) != len(groups) or not np.isfinite(v).all():
        raise ValueError("Invalid group vectors")
    return np.average(v, axis=0, weights=[len(g) for g in groups])


def view_layout(bundle, mode):
    """Reuse the same 36 comments in temporal or round-robin exposure views."""
    rows = [c for v in bundle["views"] for c in v]
    n = len(rows) // 3
    if mode == "chronological":
        return [list(range(i * n, (i + 1) * n)) for i in range(3)]
    if mode == "interleaved":
        return [list(range(i, len(rows), 3)) for i in range(3)]
    raise ValueError("Unknown view layout")


def finite_population_mean_radius(units, count, center, metric):
    """Exact mean quadratic radius for a uniform without-replacement text subset.

    Conditional on the observed finite pool, not an assumption of IID text or
    a claim that the pool mean is a psychological trait.
    """
    h = np.asarray(units, dtype=float)
    center, metric = np.asarray(center), np.asarray(metric)
    if h.ndim != 3 or not 1 <= count <= h.shape[1] or h.shape[1] < 2:
        raise ValueError("Invalid finite population or subset size")
    if center.shape != (h.shape[2],) or metric.shape != (h.shape[2], h.shape[2]):
        raise ValueError("Metric/center dimensions disagree")
    if not np.allclose(metric, metric.T) or not np.isfinite(h).all():
        raise ValueError("Need finite observations and symmetric metric")
    means = h.mean(1)
    d = means - center
    baseline = np.einsum("ni,ij,nj->n", d, metric, d)
    residual = h - means[:, None, :]
    covariance_trace = np.einsum("nti,ij,ntj->n", residual, metric, residual) / (
        h.shape[1] - 1
    )
    inflation = (1 / count - 1 / h.shape[1]) * covariance_trace
    return baseline, inflation


def linear_cka_dual(left, right):
    """Descriptive linear CKA on aligned rows, without feature-by-feature matrices."""
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 3:
        raise ValueError("CKA needs aligned observations")
    x, y = x - x.mean(0), y - y.mean(0)
    k, l = x @ x.T, y @ y.T
    denominator = np.linalg.norm(k) * np.linalg.norm(l)
    return float(np.sum(k * l) / denominator) if denominator > 1e-14 else float("nan")
