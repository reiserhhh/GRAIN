#!/usr/bin/env python3
"""Exact sufficient-statistic refits of the frozen F08 source estimators."""

import numpy as np

FEATURES = np.array([[1, b, a, a * b] for a in (0, 1) for b in (0, 1)], float)


def author_statistics(groups):
    """Keep each author's initial state and full ordered transition counts."""
    counts, initial, totals = [], [], []
    for rows in groups:
        rows = sorted(rows, key=lambda r: r["t"])
        y = np.array([r["observation"] for r in rows])
        if len(y) < 3 or not np.isin(y, [0, 1]).all():
            raise ValueError("three or more binary observations required")
        y = y.astype(int)
        c = np.zeros((4, 2))
        np.add.at(c, (2 * y[:-2] + y[1:-1], y[2:]), 1)
        first = np.zeros((2, 2))
        first[y[0], y[1]] = 1
        counts.append(c)
        initial.append(first)
        totals.append([y.sum(), len(y)])
    return {
        "counts": np.array(counts),
        "initial": np.array(initial),
        "totals": np.array(totals),
    }


def refit_statistics(stats, indices):
    """Fit a resampled author multiset; repetitions retain their multiplicity."""
    indices = np.asarray(indices, dtype=int)
    if (
        indices.ndim != 1
        or not len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(stats["counts"]))
    ):
        raise ValueError("valid nonempty source author indices required")
    individual = stats["counts"][indices]
    pooled = individual.sum(axis=0)
    tables = (individual[:, :, 1] + 1) / (individual.sum(axis=2) + 2)
    center = tables.mean(axis=0)
    _, singular, right = np.linalg.svd(tables - center, full_matrices=False)
    gram = FEATURES.T @ (pooled.sum(axis=1)[:, None] * FEATURES)
    rhs = FEATURES.T @ pooled[:, 1]
    initial = stats["initial"][indices].sum(axis=0) + 1
    total = stats["totals"][indices].sum(axis=0)
    return {
        "b1": {"source_p": (pooled[:, 1] + 1) / (pooled.sum(axis=1) + 2)},
        "b2": {"coef": np.linalg.solve(gram + np.eye(4), rhs), "alpha": 1.0},
        "m": {
            "mean": center,
            "right": right[:2],
            "singular_values": singular,
            "rank": 2,
        },
        "initial_pair": initial / initial.sum(),
        "source_mean": float(total[0] / total[1]),
    }
