#!/usr/bin/env python3
"""Empirical response-function geometry and retained shared/specific components."""

from __future__ import annotations

import numpy as np


def functional_metric(gradients, displacements, anchor):
    """Compute a within-query author covariance Gram and one fixed anchor split.

    Covectors have shape (readers, coordinates); displacements have shape
    (authors, recipient_queries, coordinates). Both channels are retained.
    """
    g, d = np.asarray(gradients, float), np.asarray(displacements, float)
    if (
        g.ndim != 2
        or d.ndim != 3
        or min(g.shape) == 0
        or min(d.shape) == 0
        or len(d) < 2
        or g.shape[1] != d.shape[2]
        or not np.isfinite(g).all()
        or not np.isfinite(d).all()
        or not isinstance(anchor, int)
        or not 0 <= anchor < len(g)
    ):
        raise ValueError(
            "Finite reference functions, author/query support and anchor required"
        )
    scores = np.einsum("rd,uqd->uqr", g, d)
    centered = scores - scores.mean(axis=0, keepdims=True)
    flat = centered.reshape(-1, len(g))
    gram = flat.T @ flat / len(flat)
    energy = np.diag(gram)
    denominator = np.sqrt(energy[:, None] * energy[None, :])
    normalized = np.divide(
        gram, denominator, out=np.full(gram.shape, np.nan), where=denominator > 0
    )
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    if gram[anchor, anchor] <= 0:
        raise ValueError("Anchor has zero reference response variance")
    coefficients = gram[:, anchor] / gram[anchor, anchor]
    shared = coefficients[:, None] * g[anchor]
    specific = g - shared
    reference_specific = np.einsum("rd,uqd->uqr", specific, d)
    residual = reference_specific - reference_specific.mean(axis=0, keepdims=True)
    orthogonality = np.mean(residual * centered[:, :, anchor, None], axis=(0, 1))
    return {
        "gram": gram,
        "normalized": normalized,
        "eigenvalues": eigenvalues,
        "coefficients": coefficients,
        "shared": shared,
        "specific": specific,
        "reference_anchor_covariance": orthogonality,
        "reference_scores": scores,
    }
