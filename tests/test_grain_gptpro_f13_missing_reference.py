#!/usr/bin/env python3
"""Missingness mechanisms must change inclusion, not rename an outcome bias."""

import numpy as np

from scripts.run_grain_gptpro_f13_missing_reference import missing_probabilities


def test_missingness_dependencies():
    """MAR uses only condition; MNAR genuinely responds to the hidden value."""
    z = np.array([-1.0, 1.0])
    y = np.array([[-2.0, 2.0], [2.0, -2.0]])
    np.testing.assert_allclose(missing_probabilities("MCAR", z, y), 0.2)
    np.testing.assert_allclose(
        missing_probabilities("MAR", z, y), [[0.2, 0.6], [0.2, 0.6]]
    )
    m = missing_probabilities("MNAR", z, y)
    assert m[0, 0] < m[1, 0] and m[0, 1] > m[1, 1]
    assert np.all((m > 0.2) & (m < 0.6))
