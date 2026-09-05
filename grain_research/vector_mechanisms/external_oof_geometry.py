"""Exact OOF score decompositions; diagnostics never used to recalibrate test outputs."""

from __future__ import annotations

import numpy as np

from .external_heads import correlation


def pearson_partition(y, p, folds):
    """Apply total covariance/variance to the frozen test-fold partition."""
    y, p, folds = np.asarray(y, float), np.asarray(p, float), np.asarray(folds)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or folds.shape != y.shape
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
        or len(y) < 3
    ):
        raise ValueError("Require aligned finite scalar scores")
    my, mp = y.mean(), p.mean()
    within_cov = between_cov = within_y = within_p = between_y = between_p = 0.0
    correlations, weights, products = [], [], []
    for f in np.unique(folds):
        ix = folds == f
        a, b = y[ix], p[ix]
        w = len(a) / len(y)
        ay, ap = a.mean(), b.mean()
        within_cov += w * np.mean((a - ay) * (b - ap))
        between_cov += w * (ay - my) * (ap - mp)
        within_y += w * np.var(a)
        within_p += w * np.var(b)
        between_y += w * (ay - my) ** 2
        between_p += w * (ap - mp) ** 2
        correlations.append(correlation(a, b))
        weights.append(w)
        products.append(w * np.std(a) * np.std(b))
    covariance = np.mean((y - my) * (p - mp))
    denominator = np.sqrt(np.var(y) * np.var(p))
    if denominator <= 1e-15:
        raise ValueError("Undefined pooled Pearson")
    total = covariance / denominator
    within_component, between_component = (
        within_cov / denominator,
        between_cov / denominator,
    )
    coefficients = np.asarray(products) / denominator
    mass = float(coefficients.sum())
    weighted_r = float(np.dot(weights, correlations))
    scale_term = (mass - 1) * weighted_r
    reweighting_term = float(
        np.dot(coefficients - mass * np.asarray(weights), correlations)
    )
    errors = [
        abs(covariance - within_cov - between_cov),
        abs(np.var(y) - within_y - between_y),
        abs(np.var(p) - within_p - between_p),
        abs(total - within_component - between_component),
    ]
    if np.isfinite(weighted_r):
        errors.append(
            abs(total - weighted_r - scale_term - reweighting_term - between_component)
        )
    if max(errors) > 1e-10:
        raise ValueError("Covariance partition failed to reconstruct")
    return {
        "pooled_pearson": float(total),
        "mean_fold_pearson": float(np.mean(correlations)),
        "sample_weighted_fold_r": weighted_r,
        "within_covariance_coefficient_mass": mass,
        "dispersion_alignment_term": float(scale_term),
        "fold_correlation_reweighting_term": reweighting_term,
        "within_covariance": float(within_cov),
        "between_covariance": float(between_cov),
        "within_r_component": float(within_component),
        "between_r_component": float(between_component),
        "within_centered_r_diagnostic_only": float(
            within_cov / np.sqrt(within_y * within_p)
        )
        if within_y * within_p > 1e-30
        else float("nan"),
        "prediction_between_variance_fraction": float(
            between_p / (within_p + between_p)
        ),
        "target_between_variance_fraction": float(between_y / (within_y + between_y)),
        "reconstruction_error": float(max(errors)),
        "not_a_new_prediction_or_test_calibration": True,
    }


def pair_auc(positive, negative):
    """Exact cross-pool AUC including half weight for tied predictions."""
    positive, negative = np.asarray(positive), np.sort(np.asarray(negative))
    if not len(positive) or not len(negative):
        return float("nan")
    less = np.searchsorted(negative, positive, side="left")
    less_equal = np.searchsorted(negative, positive, side="right")
    return float(np.sum((less + less_equal) / 2) / (len(positive) * len(negative)))


def auc_partition(y, p, folds):
    """Separate within-fold and cross-fold positive-negative ranking contributions."""
    y, p, folds = np.asarray(y), np.asarray(p, float), np.asarray(folds)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or y.shape != folds.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or len(np.unique(y)) != 2
    ):
        raise ValueError("Require aligned binary labels and finite ranking scores")
    total_pairs = int(np.sum(y == 1) * np.sum(y == 0))
    within_pairs = cross_pairs = within_weighted = cross_weighted = 0.0
    for f in np.unique(folds):
        pos = p[(folds == f) & (y == 1)]
        for g in np.unique(folds):
            neg = p[(folds == g) & (y == 0)]
            pairs = len(pos) * len(neg)
            if not pairs:
                continue
            score = pair_auc(pos, neg)
            if f == g:
                within_pairs += pairs
                within_weighted += pairs * score
            else:
                cross_pairs += pairs
                cross_weighted += pairs * score
    score = pair_auc(p[y == 1], p[y == 0])
    reconstructed = (within_weighted + cross_weighted) / total_pairs
    if within_pairs + cross_pairs != total_pairs or abs(score - reconstructed) > 1e-12:
        raise ValueError("AUC pair partition failed")
    return {
        "pooled_auc": score,
        "within_pair_fraction": within_pairs / total_pairs,
        "cross_pair_fraction": cross_pairs / total_pairs,
        "within_pair_auc": within_weighted / within_pairs
        if within_pairs
        else float("nan"),
        "cross_pair_auc": cross_weighted / cross_pairs if cross_pairs else float("nan"),
        "within_auc_component": within_weighted / total_pairs,
        "cross_auc_component": cross_weighted / total_pairs,
        "reconstruction_error": abs(score - reconstructed),
        "not_a_new_prediction_or_test_calibration": True,
    }
