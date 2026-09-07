#!/usr/bin/env python3
"""Synthetic tests of teacher-directed coordinates and frozen scoring."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from scripts.run_grain_supervised_student import (
    coordinates,
    fit_student,
    geometry,
    inner_select,
    teacher_arrays,
)


def test_pls_span_matches_standard_pls():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 16))
    y = x[:, 0] - 2 * x[:, 4] + rng.normal(size=80) * 0.1
    center, scale, basis = geometry(x, y, "pls", 4)
    reference = PLSRegression(n_components=4, scale=False).fit((x - center) / scale, y)
    q = np.linalg.qr(reference.x_rotations_)[0]
    assert np.allclose(basis @ basis.T, q @ q.T, atol=1e-8)


def test_portable_affine_scores_are_batch_independent():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(70, 12))
    y = x[:, 2] + rng.normal(size=70) * 0.01
    for method in ("full", "pca64", "pls"):
        b, a = fit_student(x[:50], y[:50], method, 4 if method == "pls" else 0, 1)
        pred = x[50:] @ b + a
        assert np.allclose(pred[:1], x[50:51] @ b + a)
        assert np.corrcoef(pred, y[50:])[0, 1] > 0.98


def test_geometry_depends_only_on_training_teacher():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(50, 8))
    g = geometry(x[:30], x[:30, 0], "pls", 2)
    z = coordinates(x[30:], g, 2)
    perturbed = x[30:].copy()
    perturbed[1:] *= 100
    assert np.allclose(z[:1], coordinates(perturbed, g, 2)[:1])


def test_inner_selection_uses_teacher_only():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(36, 10))
    rank, alpha, loss = inner_select(
        x, x[:, 0], np.arange(36).astype(str), "pls", "test"
    )
    assert rank in (2, 4, 8, 16, 32)
    assert alpha in (1, 10, 100) and loss >= 0


def test_teacher_join_rejects_missing_cache():
    import pandas as pd
    import pytest

    targets = pd.DataFrame(
        {"query": ["q", "q"], "author_id": ["a", "b"], "native_query": [1, 2]}
    )
    with pytest.raises(ValueError, match="Missing"):
        teacher_arrays(np.array(["a"]), np.zeros((1, 3)), targets, "q")
