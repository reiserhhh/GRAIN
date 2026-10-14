#!/usr/bin/env python3
"""SOURCE-oriented local response, distinct from an even repeatability witness."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_output_observability as O


def oriented_accounts(left, right, response):
    """Return signed SOURCE contact with local cross-view response strength."""
    a, b, k = (np.asarray(x, float) for x in (left, right, response))
    if (
        a.shape != b.shape
        or a.ndim < 1
        or a.shape[-1] < 2
        or k.shape != (a.shape[-1],)
        or not all(np.isfinite(x).all() for x in (a, b, k))
    ):
        raise ValueError("Two aligned finite author views and response per author")
    c = O.centered_scores((a + b) / 2)
    return np.mean(c * k, axis=-1)


def source_pullback(left_weights, right_weights, response):
    """Compile the identical linear SOURCE functional, without fitting a direction."""
    a, b, k = (np.asarray(x, float) for x in (left_weights, right_weights, response))
    if (
        a.ndim != 2
        or a.shape != b.shape
        or a.shape[0] < 2
        or a.shape[1] < 1
        or k.shape != (len(a),)
        or not all(np.isfinite(x).all() for x in (a, b, k))
    ):
        raise ValueError("Finite author-by-SOURCE pullbacks and local responses")
    d = (a + b) / 2
    return np.mean((d - d.mean(0)) * k[:, None], axis=0)


def orbit_variance(pullback, covariance):
    """Compute finite conditional variance, not a Gaussian significance claim."""
    ell, v = np.asarray(pullback, float), np.asarray(covariance, float)
    if (
        ell.ndim != 1
        or v.shape != (len(ell), len(ell))
        or not np.isfinite(ell).all()
        or not np.isfinite(v).all()
        or not np.allclose(v, v.T)
    ):
        raise ValueError("A finite SOURCE covector and symmetric orbit covariance")
    value = float(ell @ v @ ell)
    if value < 0:
        raise ArithmeticError("Negative orbit variance must not be clipped")
    return value


def cold(record):
    """Read identified numeric observations, no application psychological outcomes."""
    O.cold(record)
    scores = np.asarray(record["source_readings"], float)
    kernels = np.asarray(record["cross_output_kernels"], float)
    result = np.empty((5, 5, 4))
    for q in range(5):
        for r in range(5):
            for channel in range(4):
                result[q, r, channel] = oriented_accounts(
                    scores[0, :, q], scores[1, :, q], np.diag(kernels[channel, r])
                )
    return {
        "oriented_source_query_channels": result.tolist(),
        "source_label_reversal_changes_sign": True,
        "not_individual_trait_predictions": True,
    }
