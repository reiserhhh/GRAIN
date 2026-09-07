#!/usr/bin/env python3
"""Frozen prediction and teacher-only model-selection checks."""

import numpy as np

from scripts.run_grain_student_learning_curve import fit_model, predict, select_alpha


def test_pca_student_prediction_is_batch_independent():
    rng = np.random.default_rng(61)
    x = rng.normal(size=(40, 10))
    y = x[:, 2] + rng.normal(size=40) * 0.1
    model = fit_model(x[:30], y[:30], 1.0)
    assert np.allclose(predict(x[30:31], model), predict(x[30:], model)[:1])
    assert np.corrcoef(predict(x[30:], model), y[30:])[0, 1] > 0.9


def test_inner_selection_has_only_teacher_inputs():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(24, 8))
    y = x[:, 0]
    alpha = select_alpha(x, y, np.arange(24).astype(str), "test")
    assert alpha in (1.0, 10.0, 100.0)
