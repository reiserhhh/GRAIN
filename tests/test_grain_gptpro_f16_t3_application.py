#!/usr/bin/env python3
"""Coordinator tests for target-matched T3 application, not independent construction."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3 import f16_independent_t3 as t3
from grain_research.gptpro_foundation_v3.f16_t3_application import (
    dense_future,
    planned_outcomes,
)


def test_public_t3_facade_preserves_independent_target(monkeypatch):
    from grain_research.gptpro_foundation_v3 import f16_factory

    sentinel = {"target_id": "T3", "sentinel": "unchanged"}
    monkeypatch.setattr(t3, "cold_score", lambda raw, bundle: (raw, bundle))
    assert f16_factory.cold_score({"prefix": []}, sentinel) == (
        {"prefix": []},
        sentinel,
    )
    with pytest.raises(ValueError):
        f16_factory.build_instrument({"target_id": "T3"}, {})


def test_planned_outcome_preserves_state_and_measurement_error():
    a, b = 0.3, 0.7
    states = np.array([0.1, -0.2, 0.3, 0.6])
    errors = np.array([-0.1, 0.5, 0.1, -0.3])
    original_z = np.full(4, -1.0)
    plan = np.array([1, -1, 1, -1])
    rows = [
        {"world": "core", "y": y, "z": z}
        for y, z in zip(a + b * original_z + states + errors, original_z, strict=True)
    ]
    actual = planned_outcomes(rows, b, plan)
    np.testing.assert_allclose(actual, a + b * plan + states + errors, atol=1e-14)
    assert actual.mean() - np.mean([row["y"] for row in rows]) == pytest.approx(b)
    np.testing.assert_allclose(
        planned_outcomes(rows, b, original_z), [row["y"] for row in rows]
    )


@pytest.mark.parametrize(
    "world,slope,plan",
    [("shared_shock", 0.2, [1]), ("core", float("nan"), [1]), ("core", 0.2, [1, -1])],
)
def test_unsupported_outcome_contract_rejected(world, slope, plan):
    with pytest.raises(ValueError):
        planned_outcomes([{"world": world, "y": 0.2, "z": -1}], slope, plan)


@pytest.mark.parametrize("rho", [0.0, -0.72, 0.6, 0.98])
def test_dense_future_matches_independent_filter_and_joint_formula(rho):
    parameters = np.array([[0.2, -0.3, np.arctanh(rho / 0.99)]])
    logs = [-0.4, -0.8]
    rows = [
        {"t": t, "z": z, "y": y}
        for t, z, y in [(0, -1, 0.3), (1, 1, -0.1), (5, 1, 0.4), (7, -1, -0.2)]
    ]
    obs = [
        (0 if i == 0 else row["t"] - rows[i - 1]["t"], row["z"], row["y"])
        for i, row in enumerate(rows)
    ]
    m, p, _ = t3.filter_parameters(parameters, obs, logs)
    score, variance, _ = t3.predictive_moments(parameters, m, p, logs)
    actual = dense_future(parameters[0], rows, logs)
    np.testing.assert_allclose(actual, [score[0], variance[0]], atol=1e-12, rtol=0)


def test_planned_mean_has_slope_cancellation_but_condition_matching_is_required():
    rows = [{"world": "core", "z": -1, "y": 0.0} for _ in range(4)]
    # Reusing the historical all-minus future would evaluate a different target.
    assert planned_outcomes(rows, 2.0, [1, -1, 1, -1]).mean() == 2.0
    assert np.mean([row["y"] for row in rows]) == 0.0
