#!/usr/bin/env python3
"""Ensemble history recursion matches the existing scalar F08 implementation."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory
from grain_research.gptpro_foundation_v3.f13_binary_mixture import BinarySourceEnsemble


@pytest.mark.parametrize("complete", [False, True])
def test_scalar_equivalence(complete):
    """Check every intermediate distribution and exact Bernoulli variance."""
    fits = [
        {"b1": {"source_p": p}, "initial_pair": [[0.1, 0.2], [0.3, 0.4]]}
        for p in ([0.2, 0.7, 0.8, 0.3], [0.45, 0.5, 0.55, 0.6])
    ]
    batch = BinarySourceEnsemble(fits, complete)
    scalar = [
        QualifiedHistory(
            f, f["initial_pair"], mode="independent" if complete else "filtered"
        )
        for f in fits
    ]
    rng = np.random.default_rng(19)
    for t in range(30):
        p, q = rng.random(2)
        q = 1 if complete else q
        batch.observe(p, q)
        for i, s in enumerate(scalar):
            s.observe(p, q)
            np.testing.assert_allclose(batch.pair[i], s.pair, atol=1e-13)
        if t > 0:
            result = batch.components(rng)
            np.testing.assert_allclose(
                result["outcome_variance"],
                result["mean"] * (1 - result["mean"]),
                atol=1e-13,
            )
