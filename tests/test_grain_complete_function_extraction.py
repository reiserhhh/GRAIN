#!/usr/bin/env python3
"""Finite-difference and membership tests for frozen scalar extraction."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.complete_function_extraction import (
    apply_scalar,
    design,
    features,
    fit_scalar,
)
from grain_research.vector_mechanisms.source_state_transport import source_set
from grain_research.vector_mechanisms.whole_path_assembly import error_account


def sample():
    """Create reproducible source data with a constant column and zero final view motion."""
    r = np.random.default_rng(57)
    x, d = r.normal(size=(4, 8)), r.normal(size=(4, 4, 8))
    x[:, -1], d[:, :, 4:] = 3, 0
    return x, d, r.normal(size=4), r.normal(size=(4, 4))


@pytest.mark.parametrize("objective", ["value", "joint"])
def test_matched_normalization_and_normal_equations(objective):
    x, d, y, t = sample()
    model, diag = fit_scalar(x, d, y, t, objective, range(4))
    a, b, _, _, _, _ = design(x, d, y, t, model)
    np.testing.assert_allclose(a.T @ (a @ model["coef"] - b), 0, atol=1e-10)
    assert model["scale"][-1] == 1
    assert diag["last_direction_max"] == 0
    assert diag["source_authors"] == 4
    if objective == "value":
        # Changing derivative targets must not alter the value-only solution.
        other, _ = fit_scalar(x, d, y, t * 100, objective, range(4))
        np.testing.assert_array_equal(other["coef"], model["coef"])


def test_source_exclusion_and_matched_objectives():
    x, d, y, t = sample()
    members = source_set("source4", 2)
    models = [
        fit_scalar(x[members], d[members], y[members], t[members], o, members)[0]
        for o in ("value", "joint")
    ]
    for field in ("center", "scale", "origin", "target_scale"):
        np.testing.assert_array_equal(models[0][field], models[1][field])
    x[2], y[2], t[2] = 1e8, -1e8, 1e8
    other, _ = fit_scalar(
        x[members], d[members], y[members], t[members], "joint", members
    )
    np.testing.assert_array_equal(other["coef"], models[1]["coef"])
    assert 2 not in other["members"]


def test_full_feature_function_finite_difference():
    r = np.random.default_rng(5)
    x, d = r.normal(size=(9, 7)), r.normal(size=(4, 9, 7))
    d[:, [0, 8]] = 0
    pos, mu, u = np.arange(1, 8), r.normal(size=7), r.normal(size=(7, 4))
    phi, dp = features(x, d, pos, mu, u)
    np.testing.assert_array_equal(dp[:, 4:], 0)
    xx, dd, yy, tt = sample()
    m, _ = fit_scalar(xx, dd, yy, tt, "joint", range(4))
    _, tau = apply_scalar(phi, dp, m)
    for i in range(4):
        vals = [
            apply_scalar(features(x + s * 1e-5 * d[i], d, pos, mu, u)[0], dp, m)[0]
            for s in (1, -1)
        ]
        assert (vals[0] - vals[1]) / 2e-5 == pytest.approx(tau[i], abs=1e-8)


def test_loss_averaging_and_error_pairing():
    x, d, y, t = sample()
    model, _ = fit_scalar(x, d, y, t, "joint", range(4))
    a, b, *_ = design(x, d, y, t, model)
    yp, tp = apply_scalar(x, d, model)
    expected = np.mean(((yp - y) / model["target_scale"]) ** 2)
    expected += np.mean(((tp - t) / model["target_scale"]) ** 2)
    expected += 0.01 * np.sum(model["coef"][1:] ** 2)
    assert np.sum((a @ model["coef"] - b) ** 2) == pytest.approx(expected)
    assert error_account(y, np.zeros(4), yp)["identity_error"] < 1e-12
    with pytest.raises(ValueError):
        fit_scalar(x, d, y, t, "joint", [0, 0, 1, 2])
