"""Same-material representation comparisons, with author-node uncertainty."""

from __future__ import annotations

import numpy as np


def pair_concordance(values: np.ndarray) -> np.ndarray:
    """Return directed author-versus-impostor comparisons for three disjoint views.

    The diagonal is excluded by downstream weighting. This matches the pilot's
    earlier-to-later view convention, not a questionnaire prediction AUC.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("Expected author-by-three-views-by-features")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite representation")
    values = values / np.maximum(np.linalg.norm(values, axis=2, keepdims=True), 1e-12)
    comparisons = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        similarity = values[:, a] @ values[:, b].T
        delta = np.diag(similarity)[:, None] - similarity
        comparisons.append((delta > 1e-10) + 0.5 * (abs(delta) <= 1e-10))
    return np.stack(comparisons)


def weighted_rank(concordance: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Resample author nodes, never count repeated copies as different impostors."""
    c = np.asarray(concordance, dtype=float).mean(axis=0)
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    if c.shape != (w.shape[1], w.shape[1]) or np.any(w < 0):
        raise ValueError("Invalid author weights")
    numerator = np.einsum("bi,ij,bj->b", w, c, w) - (w * w) @ np.diag(c)
    denominator = w.sum(axis=1) ** 2 - (w * w).sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Need at least two distinct author nodes")
    return numerator / denominator


def node_bootstrap_weights(n: int, *, draws=2000, seed=731) -> np.ndarray:
    """Generate common node weights for paired method differences."""
    if n < 3:
        raise ValueError("At least three authors needed for bootstrap")
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < draws:
        counts = np.bincount(rng.integers(n, size=n), minlength=n)
        if np.count_nonzero(counts) >= 2:
            rows.append(counts)
    return np.stack(rows)


def describe_representation(
    values: np.ndarray, weights: np.ndarray
) -> tuple[dict, np.ndarray]:
    """Compute point/range and conditional author-node percentile intervals."""
    c = pair_concordance(values)
    point = float(weighted_rank(c, np.ones(len(values)))[0])
    boot = weighted_rank(c, weights)
    pair_values = [
        float(weighted_rank(c[i : i + 1], np.ones(len(values)))[0]) for i in range(3)
    ]
    return {
        "author_rank_auc": point,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "min_view_pair": min(pair_values),
        "max_view_pair": max(pair_values),
        "authors": len(values),
        "dimensions": values.shape[-1],
    }, boot


def shuffled_author_null(values: np.ndarray, *, draws=499, seed=731) -> dict:
    """Unrestricted author permutation; not a topic-matched or personality null."""
    rng = np.random.default_rng(seed)
    n = len(values)
    point = float(weighted_rank(pair_concordance(values), np.ones(n))[0])
    scores = []
    for _ in range(draws):
        shuffled = np.stack(
            [
                values[:, 0],
                values[rng.permutation(n), 1],
                values[rng.permutation(n), 2],
            ],
            axis=1,
        )
        scores.append(float(weighted_rank(pair_concordance(shuffled), np.ones(n))[0]))
    return {
        "permutation_p": (1 + sum(v >= point - 1e-12 for v in scores)) / (draws + 1),
        "null_mean": float(np.mean(scores)),
        "permutations": draws,
    }


def conditional_rank(concordance, masks, weights):
    """Average eligible comparisons with author-node resampling and fixed masks.

    This is a conditional edge estimand, not the all-author estimand above.
    Empty bootstrap samples remain NaN and their count must be reported.
    """
    c, m = np.asarray(concordance), np.asarray(masks, dtype=float).copy()
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    if c.shape != m.shape or c.shape[1:] != (w.shape[1], w.shape[1]):
        raise ValueError("Mask/author shapes disagree")
    for view in m:
        np.fill_diagonal(view, 0)
    num = np.einsum("bi,ij,bj->b", w, (c * m).sum(axis=0), w)
    den = np.einsum("bi,ij,bj->b", w, m.sum(axis=0), w)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
