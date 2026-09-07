#!/usr/bin/env python3
"""Exact covariance accounting and paired-comparison tests."""

import numpy as np

from scripts.analyze_grain_teacher_student_bridge import (
    covariance_split,
    paired_intervals,
)


def test_covariance_identity_with_imperfect_student():
    rng = np.random.default_rng(5)
    y, s, f = rng.normal(size=(3, 40))
    assert covariance_split(y, s, f)["identity_error"] < 1e-12


def test_identical_predictors_have_exact_zero_paired_delta():
    y = np.tile([0.0, 1.0], 20)
    score = y + np.linspace(-0.8, 0.8, len(y))
    result = paired_intervals(y, score, score, True, draws=40)
    assert result["delta"] == result["delta_lower"] == result["delta_upper"] == 0


def test_compression_can_preserve_or_remove_label_covariance():
    z = np.array([1.0, -1.0, 1.0, -1.0])
    e = np.array([1.0, 1.0, -1.0, -1.0])
    teacher, student = z + e, z
    preserved = covariance_split(z, teacher, student)
    removed = covariance_split(e, teacher, student)
    assert (
        preserved["student_label_covariance"] == preserved["teacher_label_covariance"]
    )
    assert preserved["residual_label_covariance"] == 0
    assert removed["student_label_covariance"] == 0
    assert removed["residual_label_covariance"] == removed["teacher_label_covariance"]
