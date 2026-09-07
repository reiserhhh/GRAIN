#!/usr/bin/env python3
"""Convert frozen scalar decoding weights into empirical forward patterns."""

from __future__ import annotations

import numpy as np


def reader_patterns(training, coefficient):
    """Return unit-score Euclidean and covariance patterns using training data only.

    The covariance pattern minimizes covariance-pseudoinverse norm on the sample
    covariance range at a fixed linear score. This empirical geometric property
    does not establish a causal interpretation of that score.
    """
    x, beta = np.asarray(training, float), np.asarray(coefficient, float)
    if x.ndim != 2 or len(x) < 2 or beta.shape != (x.shape[1],):
        raise ValueError("Training matrix and compatible coefficient required")
    if not np.isfinite(x).all() or not np.isfinite(beta).all():
        raise ValueError("Finite training geometry required")
    centered = x - x.mean(0)
    s = centered @ beta
    variance = float(s @ s / (len(x) - 1))
    norm = float(beta @ beta)
    if variance <= 0 or norm <= 0:
        raise ValueError("Nonzero coefficient and positive score variance required")
    covariance_beta = centered.T @ s / (len(x) - 1)
    patterns = {"euclidean": beta / norm, "covariance": covariance_beta / variance}
    if any(
        not np.isfinite(a).all() or not np.isclose(beta @ a, 1, atol=1e-9, rtol=1e-9)
        for a in patterns.values()
    ):
        raise ValueError("Pattern unit-score identity failed")
    return patterns, {
        "training_authors": len(x),
        "score_variance": variance,
        "beta_norm": float(np.sqrt(norm)),
    }


def split_reader_component(vector, coefficient, pattern):
    """Decompose a full pairing vector relative to zero pairing, not its mean."""
    d, beta, a = (np.asarray(x, float) for x in (vector, coefficient, pattern))
    if (
        d.ndim != 1
        or d.shape != beta.shape
        or d.shape != a.shape
        or not np.isfinite([d, beta, a]).all()
    ):
        raise ValueError("Finite compatible vectors required")
    if not np.isclose(beta @ a, 1, atol=1e-9, rtol=1e-9):
        raise ValueError("Pattern must have unit reader score")
    selected = a * (beta @ d)
    null = d - selected
    return selected, null


def effect_decomposition(native, full_removed, selected_removed, null_removed):
    """Keep interaction rather than forcing nonlinear endpoint effects to add."""
    full = native - full_removed
    selected, null = native - selected_removed, native - null_removed
    return {
        "full_effect": full,
        "selected_effect": selected,
        "null_effect": null,
        "interaction": full - selected - null,
    }
