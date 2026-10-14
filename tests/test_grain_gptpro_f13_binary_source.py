#!/usr/bin/env python3
"""Source bootstrap equivalence to original row-level estimators."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f13_binary_source import (
    author_statistics,
    refit_statistics,
)
from grain_research.measurement_discovery.estimators import fit_b1, fit_b2, fit_m


def test_repeated_authors_preserve_all_three_fits():
    """Duplicate sampled authors remain duplicate independent source copies."""
    rng = np.random.default_rng(42)
    groups = [
        [{"t": t, "observation": int(y)} for t, y in enumerate(rng.integers(0, 2, 19))]
        for _ in range(8)
    ]
    ids = np.array([0, 0, 2, 2, 2, 5, 6, 7])
    result = refit_statistics(author_statistics(groups), ids)
    copies = {str(j): groups[i] for j, i in enumerate(ids)}
    np.testing.assert_allclose(result["b1"]["source_p"], fit_b1(copies)["source_p"])
    np.testing.assert_allclose(result["b2"]["coef"], fit_b2(copies)["coef"], atol=1e-12)
    old = fit_m(copies)
    np.testing.assert_allclose(result["m"]["mean"], old["mean"])
    np.testing.assert_allclose(
        result["m"]["right"].T @ result["m"]["right"],
        old["right"].T @ old["right"],
        atol=1e-12,
    )
    assert sum(len(v) for v in copies.values()) == 152


def test_bad_source_rejected():
    """Nonbinary observations cannot silently become transition addresses."""
    with pytest.raises(ValueError):
        author_statistics([[{"t": i, "observation": i} for i in range(3)]])
    with pytest.raises(ValueError):
        author_statistics([[{"t": i, "observation": .7} for i in range(3)]])
