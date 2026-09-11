#!/usr/bin/env python3
"""Post-freeze source support and exact coordinate-contribution accounting."""

import numpy as np

from . import material_reader_field as base


def account(model, points):
    """Return fixed-kernel coverage and three source observation contributions.

    Coverage is a regularized unit-kernel Schur complement, not a calibrated
    human uncertainty estimate or a prediction residual confidence interval.
    """
    x, v, kinds = [model[k] for k in ("x", "v", "kinds")]
    row = base.gram(
        points, np.zeros_like(points), np.zeros(len(points), bool), x, v, kinds
    )
    kernel = base.gram(x, v, kinds, x, v, kinds)
    w = model["weights"]
    h = kernel * w[:, None] * w[None] + float(model["regularizer"]) * np.eye(len(x))
    cross = row * w[None]
    covered = np.sum(cross * np.linalg.solve(h, cross.T).T, axis=1)
    remainder = 1 - covered
    if remainder.min() < -1e-8 or remainder.max() > 1 + 1e-8:
        raise ValueError("Unit-kernel Schur complement outside valid range")
    distances = np.linalg.norm(points[:, None] - x[None, :8:2], axis=2)
    parts = np.stack(
        [
            row[:, model["groups"] == gi] @ model["beta"][model["groups"] == gi]
            for gi in range(3)
        ],
        axis=1,
    )
    np.testing.assert_allclose(parts.sum(axis=1), row @ model["beta"], atol=1e-10)
    return {
        "coverage_remainder": remainder,
        "nearest_distance": distances.min(axis=1),
        "max_original_kernel": np.exp(-0.5 * distances**2).max(axis=1),
        "parts": parts,
    }
