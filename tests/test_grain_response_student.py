#!/usr/bin/env python3
"""Finite-difference kernel and train-only transform tests for the S2 student."""

import numpy as np

from scripts.fit_grain_response_student import (
    fit_features,
    fit_kernel_student,
    predict_kernel_student,
    select_specs,
    transform_features,
    value_difference_kernel,
)


def test_difference_kernel_equals_explicit_feature_augmentation():
    rng = np.random.default_rng(1)
    x, e = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    design = np.vstack([x, e - x])
    gram = value_difference_kernel(x @ x.T, x @ e.T, e @ e.T)
    assert np.allclose(gram, design @ design.T)
    assert np.linalg.eigvalsh(gram).min() > -1e-10


def test_shared_difference_student_matches_weighted_primal_ridge():
    rng = np.random.default_rng(7)
    x, e = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
    q = np.arange(30) % 3
    y, ye = rng.normal(size=30), rng.normal(size=30)
    model = fit_kernel_student(x, e, q, y, ye, eta=10.0, gamma=0.0, ridge=1.0)
    design = np.vstack([x, np.sqrt(10) * (e - x)])
    target = np.concatenate(
        [
            (y - model["center"][q]) / model["scale"][q],
            np.sqrt(10) * (ye - y) / model["scale"][q],
        ]
    )
    coef = np.linalg.solve(design.T @ design + np.eye(4), design.T @ target)
    expected = (x @ coef) * model["scale"][q] + model["center"][q]
    assert np.allclose(predict_kernel_student(x, q, model), expected)


def test_frozen_feature_transforms_do_not_depend_on_other_test_rows():
    rng = np.random.default_rng(9)
    comments = [rng.normal(size=(8, 12)) for _ in range(12)]
    q, poles = np.arange(12) % 3, rng.normal(size=(3, 2, 12))
    for family in ("mean", "query", "distribution", "quadratic"):
        spec = fit_features(comments[:9], q[:9], family, poles)
        one = transform_features(comments[9:10], q[9:10], spec)
        batch = transform_features(comments[9:], q[9:], spec)
        assert np.allclose(one[0], batch[0])
        assert np.isfinite(batch).all()


def test_selection_does_not_read_unfitted_edit_targets():
    rng = np.random.default_rng(41)
    groups = np.repeat([f"a{j}" for j in range(9)], 3)
    q = np.arange(len(groups)) % 3
    comments = [[rng.normal(size=(5, 4)) for _ in range(4)] for _ in groups]
    values = rng.normal(size=(len(groups), 4))
    poles = rng.normal(size=(3, 2, 4))
    before = select_specs(comments, values, q, groups, "mean", poles, "test")
    values[:, 2:] = np.nan
    after = select_specs(comments, values, q, groups, "mean", poles, "test")
    assert before == after
