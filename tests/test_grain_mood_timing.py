#!/usr/bin/env python3
"""Partial correlation agrees with residual projection without refitting a predictor."""

import numpy as np

from scripts.analyze_grain_mood_timing import partial_correlation


def test_partial_r_matches_residual_geometry():
    rng = np.random.default_rng(3)
    z, a, b = rng.normal(size=(3, 60))
    x, y = z + a, 2 * z + 0.4 * a + b
    result = partial_correlation(x, y, z)
    design = np.column_stack([np.ones(len(z)), z])
    residuals = (
        np.column_stack([x, y])
        - design @ np.linalg.lstsq(design, np.column_stack([x, y]), rcond=None)[0]
    )
    np.testing.assert_allclose(
        result["partial_r_after_given_before"],
        np.corrcoef(residuals.T)[0, 1],
        atol=1e-12,
    )
    assert result["partial_df"] == 57
    assert 0 <= result["partial_p"] <= 1
