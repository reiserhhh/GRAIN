#!/usr/bin/env python3
"""Algebra and source-only tests; no private data or model downloads required."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.conditional_models import predict_response_kernel
from grain_research.vector_mechanisms.reference_integration import (
    centered_cross_kernel,
    condition_profile,
    fit_anchor_response,
    fit_condition_reference,
    predict_anchor_response,
    selection_response_decomposition,
)


def fixture_arrays():
    """Return fixed independent rows with a few paired source conditions."""
    rng = np.random.default_rng(22)
    x, y = rng.normal(size=(24, 18)), rng.normal(size=(24, 3))
    return x @ x.T, y, np.arange(20), [(6, 7), (8, 9), (10, 11), (18, 22)]


def test_source_anchors_do_not_drift():
    gram, y, ii, pairs = fixture_arrays()
    m = fit_anchor_response(gram, y, ii, pairs, [0, 1, 2, 3])
    np.testing.assert_allclose(
        predict_anchor_response(m, gram[:4]),
        predict_response_kernel(m, gram[:4]),
        atol=1e-10,
    )
    assert float(m["projected_min_eigenvalue"]) > -1e-10
    assert 0 <= float(m["projected_trace_ratio"]) <= 1


def test_held_target_anchor_and_edges_are_excluded():
    gram, y, ii, pairs = fixture_arrays()
    m = fit_anchor_response(gram, y, ii, pairs, [0, 1, 21])
    y[20:] = np.nan
    n = fit_anchor_response(gram, y, ii, pairs, [0, 1, 21])
    np.testing.assert_array_equal(m["correction_coef"], n["correction_coef"])
    assert len(m["anchor_positions"]) == 2


def test_unseen_batch_does_not_recenter():
    gram, y, ii, pairs = fixture_arrays()
    m = fit_anchor_response(gram, y, ii, pairs, [0, 1])
    np.testing.assert_allclose(
        predict_anchor_response(m, gram[20:21]),
        predict_anchor_response(m, gram[20:])[0:1],
        atol=1e-12,
    )


def test_projection_matches_constrained_primal():
    gram, y, ii, pairs = fixture_arrays()
    m = fit_anchor_response(gram, y, ii, pairs, [0, 1])
    k = centered_cross_kernel(m, gram[ii])
    eigen, vec = np.linalg.eigh(k)
    select = eigen > 1e-10
    phi = vec[:, select] * np.sqrt(eigen[select])
    a = phi[:2]
    projection = np.eye(phi.shape[1]) - np.linalg.pinv(a) @ a
    xp = phi @ projection
    d = np.zeros((3, 20))
    for row, (lo, hi) in enumerate(pairs[:3]):
        d[row, lo], d[row, hi] = -1, 1
    metric = np.eye(20) + float(m["response_lambda"]) * 20 / 3 * d.T @ d
    r = y[ii] - predict_response_kernel(m, gram[ii])
    beta = np.linalg.solve(xp.T @ metric @ xp + np.eye(xp.shape[1]), xp.T @ metric @ r)
    np.testing.assert_allclose(
        predict_anchor_response(m, gram[ii]) - predict_response_kernel(m, gram[ii]),
        xp @ beta,
        atol=1e-9,
    )


def test_reference_reconstructs_and_is_batch_invariant():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(12, 8, 3))
    ref = fit_condition_reference(x[:6], np.ones(8) / 8, reference_id="source-v1")
    p, one = condition_profile(x[6:], ref), condition_profile(x[6:7], ref)
    assert p["reconstruction_max_error"] < 1e-12
    assert p["shape_weighted_mean_max"] < 1e-12
    np.testing.assert_allclose(p["standardized_level"][:1], one["standardized_level"])
    source = condition_profile(x[:6], ref)
    np.testing.assert_allclose(source["level"].mean(0), 0, atol=1e-12)


def test_constant_reference_is_not_fake_unit_variance():
    x = np.ones((3, 4, 1))
    ref = fit_condition_reference(x, np.ones(4) / 4, reference_id="constant")
    p = condition_profile(x, ref)
    assert not p["standardization_supported"][0]
    assert np.isnan(p["standardized_level"]).all()
    assert p["reconstruction_max_error"] == 0


def test_choice_response_identity_keeps_interaction():
    m, b = np.array([[2.0, 4], [8, 6]]), np.array([[1.0, 1], [3, 2]])
    p = selection_response_decomposition(m, b, [1, 0], [0.5, 0.5])
    assert p["reconstruction_error"] < 1e-12
    assert np.linalg.norm(p["interaction"]) > 0
    same = selection_response_decomposition(m, b, [0.5, 0.5], [0.5, 0.5])
    np.testing.assert_allclose(same["choice_or_exposure"], 0)
    np.testing.assert_allclose(same["interaction"], 0)


@pytest.mark.parametrize("weights", [[1, 1], [-1, 2], [np.nan, 1]])
def test_invalid_opportunity_measure_rejected(weights):
    with pytest.raises(ValueError):
        fit_condition_reference(np.ones((3, 2, 1)), weights, reference_id="r")


def test_no_retained_anchor_rejected():
    gram, y, ii, pairs = fixture_arrays()
    with pytest.raises(ValueError):
        fit_anchor_response(gram, y, ii, pairs, [22])
