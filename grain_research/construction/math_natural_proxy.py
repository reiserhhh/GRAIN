#!/usr/bin/env python3
"""Three-view linear moment construction with explicit event qualifications."""

from __future__ import annotations

import numpy as np


def view_features(coordinates, conditions, units):
    """Preserve choice and conditional vector masses in disjoint event views."""
    z, c, ids = map(np.asarray, (coordinates, conditions, units))
    if (
        z.ndim != 3
        or z.shape[1] % 3
        or c.shape != z.shape[:2]
        or ids.shape != c.shape
        or not np.isfinite(z).all()
        or not np.isin(c, [0, 1]).all()
        or any(len(np.unique(row)) != len(row) for row in ids)
    ):
        raise ValueError("Finite coordinates and distinct per-author events required")
    rows = []
    for v in range(3):
        zv, cv = z[:, v::3], c[:, v::3, None]
        rows.append(
            np.concatenate(
                [
                    np.ones((len(z), 1)),
                    cv.mean(axis=1),
                    ((1 - cv) * zv).mean(axis=1),
                    (cv * zv).mean(axis=1),
                ],
                axis=1,
            )
        )
    return np.stack(rows, axis=1)


def author_fold(folds, calibration_fold):
    """Return complementary source/application author rows, never text splits."""
    f = np.asarray(folds)
    if f.ndim != 1 or set(np.unique(f)) != {0, 1} or calibration_fold not in (0, 1):
        raise ValueError("Two complete historical author folds required")
    train, held = (
        np.flatnonzero(f == calibration_fold),
        np.flatnonzero(f != calibration_fold),
    )
    if len(train) < 2 or len(held) < 2:
        raise ValueError("Insufficient author support")
    return train, held


def construct(x, y, w, z, rank_tolerance=1e-12):
    """Compile source-only IV moments and ordinary regression without tuning."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (
        x.ndim != 3
        or x.shape[1] != 3
        or y.ndim != 2
        or len(x) != len(y)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or w == z
        or w not in (0, 1, 2)
        or z not in (0, 1, 2)
    ):
        raise ValueError("Aligned finite source views and distinct instrument required")
    n = len(x)
    k, m = x[:, z].T @ x[:, w] / n, x[:, z].T @ y / n
    bridge, _, rank, singular = np.linalg.lstsq(k, m, rcond=rank_tolerance)
    ordinary = np.linalg.lstsq(x[:, w], y, rcond=rank_tolerance)[0]
    return {
        "bridge": bridge.tolist(),
        "ordinary": ordinary.tolist(),
        "kernel": k.tolist(),
        "moment": m.tolist(),
        "rank": int(rank),
        "singular_values": singular.tolist(),
        "source_rms": np.sqrt((x * x).mean(axis=0)).tolist(),
        "equation_max_error": np.max(np.abs(k @ bridge - m), axis=0).tolist(),
        "coefficient_norm": np.linalg.norm(bridge, axis=0).tolist(),
        "ordinary_coefficient_norm": np.linalg.norm(ordinary, axis=0).tolist(),
    }


def moment_error(x, y, beta, w, instrument, source_rms):
    """Report unused/held orthogonality without an unjustified J-test p-value."""
    residual = np.asarray(y) - np.asarray(x)[:, w] @ np.asarray(beta)
    moment = np.asarray(x)[:, instrument].T @ residual / len(x)
    scale = np.asarray(source_rms)[instrument]
    normalized = np.divide(
        moment, scale[:, None], out=np.zeros_like(moment), where=scale[:, None] > 0
    )
    return {
        "raw": moment.tolist(),
        "normalized_rms": np.sqrt((normalized * normalized).mean(axis=0)).tolist(),
        "calibration_bias": residual.mean(axis=0).tolist(),
    }


def correlation(a, b):
    """Compute a signed correlation, retaining undefined constants as missing."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else None


def auc(y, score):
    """Pairwise binary AUC with half credit for ties; not threshold accuracy."""
    y, score = np.asarray(y), np.asarray(score)
    a, b = score[y == 1], score[y == 0]
    if not len(a) or not len(b):
        return None
    differences = a[:, None] - b
    return float(
        (np.sum(differences > 0) + 0.5 * np.sum(differences == 0)) / differences.size
    )


def matched_donors(counts, lengths, source_length_median):
    """Deterministically cycle strangers in registered opportunity/length cells."""
    counts, lengths = np.asarray(counts), np.asarray(lengths)
    if counts.shape != lengths.shape or counts.ndim != 1:
        raise ValueError("Aligned author-level count and length required")
    group = np.minimum(counts.astype(int) // 4, 3) * 2 + (
        lengths > source_length_median
    )
    donors = np.full(len(counts), -1, int)
    for key in np.unique(group):
        rows = np.flatnonzero(group == key)
        if len(rows) > 1:
            donors[rows] = np.roll(rows, 1)
    return donors


def permute_profiles(y, folds, rng):
    """Shuffle whole criteria profiles within historical author folds."""
    shuffled = np.asarray(y).copy()
    for f in (0, 1):
        rows = np.flatnonzero(np.asarray(folds) == f)
        shuffled[rows] = shuffled[rng.permutation(rows)]
    return shuffled
