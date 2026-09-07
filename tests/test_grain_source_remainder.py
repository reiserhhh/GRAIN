#!/usr/bin/env python3
"""Exact source-covariance identities without observer or psychological conflation."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_grain_source_remainder import component_frame, covariance


def test_three_part_identity_and_covariance_are_exact():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        rng.normal(size=(40, 4)),
        columns=["old_teacher", "new_teacher", "mean_score", "student"],
    )
    result = component_frame(frame)
    y = rng.normal(size=40)
    components = ["observer_difference", "configuration_response", "mean_remainder"]
    assert np.allclose(result[components].sum(axis=1), result.total_remainder)
    assert np.isclose(
        sum(covariance(y, result[c]) for c in components),
        covariance(y, result.total_remainder),
    )


def test_observer_difference_not_implicitly_assigned_to_configuration():
    frame = pd.DataFrame(
        {
            "old_teacher": [1.0, 2.0, 3.0],
            "new_teacher": [0.5, 1.0, 1.5],
            "mean_score": [0.5, 1.0, 1.5],
            "student": [0.5, 1.0, 1.5],
        }
    )
    result = component_frame(frame)
    assert np.allclose(result.configuration_response, 0)
    assert np.allclose(result.observer_difference, result.total_remainder)


def test_covariance_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        covariance(np.array([1.0, np.nan]), np.array([0.0, 1.0]))
