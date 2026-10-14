#!/usr/bin/env python3
"""Uncertainty fields cannot erase negative covariance, null support or failure."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f13_contract import (
    calibrate_intervals,
    measurement_result,
    uncertainty_components,
)


def test_correlated_components():
    """Negative cross covariance is real, not an invalid negative variance."""
    out = uncertainty_components(["person", "state"], [[1.0, -0.4], [-0.4, 1.0]])
    np.testing.assert_allclose(out["total_variance"], 1.2)
    assert out["cross_covariances"][0]["contribution_to_total"] == -0.8
    with pytest.raises(ValueError):
        uncertainty_components(["person", "state"], [[1.0, 2.0], [2.0, 1.0]])


def test_calibration_failure_survives_interface():
    """A numerical score remains usable without relabeling failed calibration."""
    scope = {"target": "delta", "world": "ar06", "rho": 0}
    check = calibrate_intervals(
        np.ones(200), np.zeros(200), np.full(200, 0.1), list(range(200)), scope
    )
    assert check["decision"] == "FAIL"
    result = measurement_result(0.05, 0.01, scope, "observed", check)
    assert result["calibration_check"] == "FAIL" and result["score"] == 0.05
    with pytest.raises(ValueError, match="scope"):
        measurement_result(0.05, 0.01, scope | {"target": "level"}, "observed", check)


def test_no_fake_independent_units_or_zero():
    """Repeated author rows do not increase calibration sample size."""
    with pytest.raises(ValueError, match="repeated"):
        calibrate_intervals([0, 0], [-1, -1], [1, 1], ["a", "a"], {"target": "x"})
    result = measurement_result(None, None, {"target": "x"}, "unobserved")
    assert result["score"] is None and result["calibration_check"] == "NOT_EVALUATED"
    with pytest.raises(ValueError):
        measurement_result(0.0, 0.0, {"target": "x"}, "unobserved")
    assert (
        measurement_result(0.5, 0.2, {"target": "x"}, "model_borrowed")["support"]
        == "model_borrowed"
    )
