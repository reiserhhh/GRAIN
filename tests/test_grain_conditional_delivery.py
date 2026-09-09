#!/usr/bin/env python3
"""Finite counterexamples clarify which mechanisms the conditional objective identifies."""

import numpy as np

from scripts.summarize_grain_conditional_delivery import graph_geometry


def test_edit_graph_leaves_author_offsets_invisible():
    """The measured pair graph has48 response directions and108 null directions."""
    result = graph_geometry()
    assert result["rank"] == 48
    assert result["nullity"] == 108
    assert result["intercept_error"] == 0


def test_perfect_responses_do_not_determine_author_order():
    """Exact within-author changes coexist with zero between-author association."""
    y = np.linspace(-1, 1, 12)
    z = y**2 - (y**2).mean()
    d = np.linspace(0.1, 0.8, 12)
    true_original, true_edited = y, y + d
    predicted_original, predicted_edited = z, z + d
    np.testing.assert_allclose(
        predicted_edited - predicted_original, true_edited - true_original, atol=1e-14
    )
    assert abs(np.corrcoef(true_original, predicted_original)[0, 1]) < 1e-14
