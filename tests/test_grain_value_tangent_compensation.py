#!/usr/bin/env python3
"""Compensation, source-span and unchanged scalar-function witnesses."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.complete_function_extraction import apply_scalar
from grain_research.vector_mechanisms.value_tangent_compensation import (
    author_centered,
    compensation_summary,
    coverage,
    observed_basis,
    row_space,
    score_parts,
)


def test_exact_compensation_and_amplification():
    b = np.asarray([-3.0, -1.0, 1.0, 3.0])
    a = np.column_stack((np.zeros(4), b, -b))
    s = compensation_summary(a, np.arange(4.0))
    assert s["cancellation_ratio"] == 0 and s["body_query_r"] == pytest.approx(-1)
    assert s["mse_identity_error"] < 1e-12
    a[:, 2] = b
    s = compensation_summary(a, np.arange(4.0))
    assert s["cancellation_ratio"] == pytest.approx(2)
    assert s["mse_identity_error"] < 1e-12


def test_uncorrelated_and_undefined_compensation():
    a = np.asarray([[0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], float)
    assert compensation_summary(a, np.zeros(4))["cancellation_ratio"] == 1
    assert np.isnan(
        compensation_summary(np.zeros((4, 3)), np.zeros(4))["cancellation_ratio"]
    )


def test_unseen_and_zero_rows():
    b, _, _ = observed_basis(np.asarray([[1.0, 0, 0], [2.0, 0, 0]]))
    c = coverage(np.asarray([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0]]), b)
    np.testing.assert_allclose(c[:2], [1, 0])
    assert np.isnan(c[2])


def test_joint_orthogonal_coordinates_preserve_coverage():
    rng = np.random.default_rng(58)
    a, r = rng.normal(size=(5, 12)), rng.normal(size=(7, 12))
    o, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    b, *_ = observed_basis(a)
    rotated, *_ = observed_basis(a @ o)
    np.testing.assert_allclose(coverage(r, b), coverage(r @ o, rotated), atol=1e-12)


def test_compatible_score_and_derivative_parts():
    rng = np.random.default_rng(19)
    v = {
        "center": rng.normal(size=6),
        "scale": np.ones(6),
        "origin": np.asarray(3.0),
        "target_scale": np.asarray(2.0),
        "members": np.arange(4),
        "coef": rng.normal(size=7),
    }
    j = v | {"coef": rng.normal(size=7)}
    x, d = rng.normal(size=6), rng.normal(size=(4, 6))
    d[:, 3:] = 0
    parts, dt = score_parts(x, d, v, j)
    yp, tp = apply_scalar(x, d, v)
    yj, tj = apply_scalar(x, d, j)
    assert parts.sum() == pytest.approx(yj - yp)
    np.testing.assert_allclose(dt, tj - tp)
    with pytest.raises(ValueError):
        score_parts(x, d, v, j | {"origin": np.asarray(4.0)})


def test_data_design_excludes_penalty_and_common_action_level():
    x = np.zeros((4, 6))
    d = np.zeros((4, 4, 6))
    m = {"center": np.zeros(6), "scale": np.ones(6)}
    _, diag = row_space(x, d, m)
    assert diag["rank"] == 1 and diag["nullity"] == 6
    a = np.arange(16.0).reshape(4, 4)
    np.testing.assert_array_equal(
        author_centered(a), author_centered(a + np.arange(4)[:, None] * 9)
    )
