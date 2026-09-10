#!/usr/bin/env python3
"""Finite witnesses for source-only value/gradient dictionary diagnostics."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.token_objective_compatibility import (
    diagnose,
    project,
    reconstruct_design,
    svd_operator,
)
from grain_research.vector_mechanisms.token_potential import (
    ARMS,
    fit_potential,
    source_descriptor,
    state_design,
)


def solve(v, d, ell, k, alpha):
    """Construct the exact quadratic optimum for a small synthetic witness."""
    penalty = np.eye(v.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(v.T @ v + d.T @ d + penalty, v.T @ ell + d.T @ k)


@pytest.mark.parametrize("arm", ARMS)
def test_independent_source_design_and_force_sum(arm):
    r = np.random.default_rng(610)
    hs = [r.normal(size=(t, 7)) for t in (18, 25, 32)]
    gs = [r.normal(size=h.shape) / len(h) for h in hs]
    lasts = [r.normal(size=7) for _ in hs]
    scores = np.array([1, -2, 3.0])
    desc = source_descriptor(hs, gs, [0, 2, 3], rank=3, landmarks=4)
    model, _ = fit_potential(hs, lasts, gs, scores, desc, arm)
    v, d, ell, k, anchor, errors = reconstruct_design(hs, lasts, gs, scores, model)
    assert max(errors.values()) < 1e-10
    expected = np.stack(
        [state_design(h, last, model, arm)[0] for h, last in zip(hs, lasts)]
    )
    np.testing.assert_allclose(v, expected / (np.sqrt(3) * scores.std()), atol=1e-12)
    report, packet = diagnose(v, d, ell, k, model["coef"], 0.01, anchor)
    assert report["stationarity_error"] < 1e-10
    assert report["preserved_value_change"] < 1e-8
    assert report["preserving_objective_change"] >= -1e-8
    assert report["objective_identity_relative_error"] < 1e-7
    assert report["dictionary_external_loss"] <= report["source_field_loss"] + 1e-8
    assert report["anchor_rank"] == 3
    np.testing.assert_allclose(packet["forces"].sum(0), 0, atol=1e-10)


def test_external_residual_and_free_score_anchor():
    v = np.array([[1.0, 0.0]])
    d = np.array([[0.0, 1.0], [0.0, 0.0]])
    ell, k = np.array([3.0]), np.array([1.0, 2.0])
    theta = solve(v, d, ell, k, 0.1)
    report, _ = diagnose(v, d, ell, k, theta, 0.1, [0])
    assert report["dictionary_external_loss"] == pytest.approx(4)
    assert report["preserved_score_field_loss"] == pytest.approx(4)
    assert report["field_anchor_residual"] < 1e-12
    assert report["preserved_value_change"] < 1e-12
    assert report["anchor_rank"] == 1


def test_true_value_field_conflict_has_no_free_direction():
    v = np.array([[0.0, 1.0]])
    d = np.array([[0.0, 1.0]])
    ell, k = np.array([-1.0]), np.array([1.0])
    theta = np.zeros(2)
    report, _ = diagnose(v, d, ell, k, theta, 0.1, [0])
    assert report["dictionary_external_loss"] == 0
    assert report["source_field_loss"] == 1
    assert report["preserved_score_field_gain"] == 0
    assert report["anchor_rank"] == 0
    assert report["force_value_field_inner"] < 0


def test_fixed_cutoff_and_large_inverse_amplification():
    op = svd_operator(np.diag([1.0, 1e-8, 1e-12]))
    coef, pred = project(op, np.ones(3))
    assert len(op[1]) == 2
    np.testing.assert_allclose(coef, [1, 1e8, 0])
    np.testing.assert_allclose(pred, [1, 1, 0])


def test_reject_nonzero_body_anchor():
    with pytest.raises(ValueError, match="zero BODY"):
        diagnose(np.eye(2), np.eye(2), np.ones(2), np.ones(2), np.zeros(2), 0.01, [0])


def test_no_retained_direction_is_well_defined():
    op = svd_operator(np.zeros((5, 3)))
    coef, pred = project(op, np.arange(5.0))
    np.testing.assert_array_equal(coef, np.zeros(3))
    np.testing.assert_array_equal(pred, np.zeros(5))
