#!/usr/bin/env python3
"""Synthetic source isolation, whole-cell prediction and completion algebra."""

import numpy as np
import pytest

from grain_research.conditional_response import (
    complete_functional,
    condition_kernel,
    evaluate_hidden_cells,
    fit_response_reference,
    observed_cells,
    predict_cells,
)


def material():
    """Return six authors crossing four conditions with repeated measurements."""
    rng = np.random.default_rng(81)
    a = np.repeat(np.arange(6), 12)
    c = np.tile(np.repeat(["a", "b", "c", "d"], 3), 6)
    x = rng.normal(size=(72, 3)) + a[:, None] * 0.1
    g = rng.normal(size=(72, 4))
    return c, x, g, a


def test_source_isolation():
    c, x, g, a = material()
    ref = fit_response_reference(c, x, g, a, np.arange(4))
    x[a >= 4] += 1000
    g[a >= 4] *= -100
    other = fit_response_reference(c, x, g, a, np.arange(4))
    for key in ("mean", "descriptor", "noise", "weights", "common5_weights"):
        np.testing.assert_array_equal(ref[key], other[key])
    assert ref["noise_dof"] == 32


def test_kernel_psd_permutation_and_zero():
    c, x, g, a = material()
    ref = fit_response_reference(c, x, g, a, np.arange(4))
    k = condition_kernel(ref, 0.1, 1)
    assert np.linalg.eigvalsh(k).min() > 0
    p = np.array([3, 1, 0, 2])
    np.testing.assert_allclose(condition_kernel(ref, 0.1, 1, p), k[np.ix_(p, p)])
    assert not condition_kernel(ref, 0, 0).any()
    with pytest.raises(ValueError):
        condition_kernel(ref, 1, 1, [0, 0, 2, 3])


def test_random_intercept_closed_form():
    c, x, g, a = material()
    ref = fit_response_reference(c, x, g, a, np.arange(4))
    k = condition_kernel(ref, 0.1, 0)
    idx, n = np.array([0, 2]), np.array([3, 4])
    delta = np.array([[0.2, -0.4, 0.1], [0.5, 0.1, -0.2]])
    prior, d, v = predict_cells(ref, k, idx, ref["mean"][idx] + delta, n, [1])
    expected = (n @ delta) / (10 + n.sum())
    np.testing.assert_allclose(d[0], expected)
    np.testing.assert_allclose(v[0], ref["noise"] / (10 + n.sum()))
    np.testing.assert_array_equal(prior, ref["mean"][[1]])


def test_loco_hides_entire_cell():
    c, x, g, a = material()
    ref = fit_response_reference(c, x, g, a, np.arange(4))
    k = condition_kernel(ref, 1, 0.1)
    rows, y, p, d, _ = evaluate_hidden_cells(c, x, a, [4], ref, k)
    x[(a == 4) & (c == "a")] += 100
    r2, y2, p2, d2, _ = evaluate_hidden_cells(c, x, a, [4], ref, k)
    assert rows == r2
    np.testing.assert_allclose((p + d)[0], (p2 + d2)[0])
    np.testing.assert_allclose(y2[0] - y[0], 100)


def test_completion_equals_explicit_posterior_and_keeps_observed():
    c, x, g, a = material()
    ref = fit_response_reference(c, x, g, a, np.arange(4))
    k = condition_kernel(ref, 1, 0.1)
    s, t, n = np.array([0, 2]), np.array([1, 3]), np.array([3, 3])
    means = ref["mean"][s] + 0.2
    nu = ref["weights"]
    out = complete_functional(ref, k, s, means, n, nu)
    p, d, _ = predict_cells(ref, k, s, means, n, t)
    np.testing.assert_allclose(out["model_completed"], nu[s] @ means + nu[t] @ (p + d))
    cov = k[np.ix_(t, t)] - k[np.ix_(t, s)] @ np.linalg.solve(
        k[np.ix_(s, s)] + np.diag(1 / n), k[np.ix_(s, t)]
    )
    np.testing.assert_allclose(
        out["model_latent_sd"] ** 2, (nu[t] @ cov @ nu[t]) * ref["noise"]
    )
    assert out["empirical_point"] is None


def test_full_empty_and_unknown_limits():
    c, x, g, a = material()
    ref = fit_response_reference(c, x, g, a, np.arange(4))
    k = condition_kernel(ref, 1, 1)
    cells = observed_cells(c[a == 5], x[a == 5], ref)
    full = complete_functional(
        ref, k, cells["indices"], cells["means"], cells["counts"], ref["weights"]
    )
    np.testing.assert_allclose(full["empirical_point"], full["model_completed"])
    assert not full["model_latent_sd"].any()
    empty = observed_cells(["outside"], [[1, 2, 3]], ref)
    assert empty["unknown_units"] == 1 and empty["unknown_cells"] == 1
    out = complete_functional(
        ref, k, empty["indices"], empty["means"], empty["counts"], ref["weights"]
    )
    np.testing.assert_allclose(out["model_completed"], ref["weights"] @ ref["mean"])
    assert not out["missing_personal_adjustment"].any()
    assert out["empirical_point"] is None


def test_invalid_or_self_target_rejected():
    c, x, g, a = material()
    ref = fit_response_reference(c, x, g, a, np.arange(4))
    k = condition_kernel(ref, 1, 1)
    for idx, counts, target in [
        ([0], [1], [0]),
        ([0], [0], [1]),
        ([0, 0], [1, 1], [1]),
    ]:
        with pytest.raises(ValueError):
            predict_cells(ref, k, idx, ref["mean"][idx], counts, target)


def test_constant_single_condition_boundary():
    c, x, g, a = material()
    c[:] = "one"
    ref = fit_response_reference(c, x, g, a, np.arange(4))
    np.testing.assert_array_equal(
        condition_kernel(ref, 0.1, 1), condition_kernel(ref, 1, 0.1)
    )
    _, _, _, d, _ = evaluate_hidden_cells(
        c, x, a, [4, 5], ref, condition_kernel(ref, 1, 1)
    )
    assert not d.any()
