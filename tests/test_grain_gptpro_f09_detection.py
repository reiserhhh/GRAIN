"""Calibration and grouped-count likelihood checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f09_detection import (
    calibrate,
    count_predictions,
    discrepancy,
    score_oof,
)
from grain_research.gptpro_foundation_v3.f09_rivals import (
    EVEN,
    ODD,
    fit_family,
    predict_family,
)


def test_grouped_count_posterior_matches_events():
    counts = np.array([0, 1, 2, 3])
    z = np.repeat(EVEN, 3, axis=0)
    y = np.concatenate([np.r_[np.ones(k), np.zeros(3 - k)] for k in counts])
    np.testing.assert_allclose(
        count_predictions(counts), predict_family(fit_family(z, y), ODD), atol=1e-14
    )


def test_deviance_endpoints_are_finite():
    np.testing.assert_allclose(
        discrepancy(np.array([0, 3, 0, 3]), np.full(4, 0.5)), 24 * np.log(2)
    )


def test_composite_survival_uses_max_and_finite_sample_correction():
    calibration = calibrate(3, 20)
    out = score_oof([0, 1, 2, 3], [3, 0, 1, 2], calibration["statistics"])
    expected = [
        (1 + sum(x >= out["statistic"] for x in row)) / 21
        for row in calibration["statistics"]
    ]
    np.testing.assert_allclose(out["p_by_atom"], expected)
    assert out["p_composite"] == max(expected)
