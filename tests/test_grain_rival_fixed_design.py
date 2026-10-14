"""Synthetic tests for GP04's fixed-condition rival models."""

import numpy as np

from grain_research.measurement_discovery.rival_models import (
    CORNERS,
    posterior_predictive,
    response_probability,
)


def test_corner_response_is_fixed_and_bounded():
    assert len(CORNERS) == 8
    assert response_probability("M1_axis_1", 1.0, (1, 1, 1)) > 0.5
    assert response_probability("M1_axis_1", 1.0, (-1, 1, 1)) < 0.5
    assert 0 < response_probability("M4_mixed", 1.0, (1, -1, 1)) < 1


def test_uniform_posterior_is_finite():
    candidates = [("M1_axis_1", .5), ("M2_axis_2", 1.0)]
    p, weights = posterior_predictive([], candidates, (1, -1, 1))
    assert np.isclose(weights.sum(), 1)
    assert 0 < p < 1
