#!/usr/bin/env python3
"""Post-outcome algebra check: whitening four independent samples creates a simplex."""

import numpy as np
import pytest

from scripts.report_grain_context_modulated_potential import geometry_account


def test_full_rank_whitening_erases_source_distance_variation():
    r = np.random.default_rng(6401)
    x = r.normal(size=(4, 9)) * np.arange(1, 10)
    centered = x - x.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    c = centered @ vt[:3].T
    c /= c.std(0)
    a = geometry_account(x, c)
    assert a["simplex_gram_error"] < 1e-12
    assert a["standardized_pair_distance_cv"] < 1e-12
    assert a["raw_pair_distance_cv"] > 0.01
    assert a["standardized_pair_distance_min"] == pytest.approx(np.sqrt(8))


def test_rank_deficient_sources_do_not_satisfy_full_simplex_premise():
    with pytest.raises(ValueError, match="Full centered sample rank"):
        geometry_account(np.zeros((4, 9)), np.zeros((4, 3)))
