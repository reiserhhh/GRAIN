#!/usr/bin/env python3
"""Compilation equals the declared source operator with anisotropic normalization."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.compiled_gated_readout import (
    mean_readout,
    source_readout,
)
from grain_research.vector_mechanisms.gated_update import (
    alternatives,
    apply_coefficients,
    coefficients,
    rms_direction,
)


@pytest.mark.parametrize("epsilon", [1e-6, 0.2])
def test_adjoint_compilation_and_paired_mean(epsilon):
    rng = np.random.default_rng(44)
    h, k = 7, 11
    weights = {
        "gate": rng.normal(size=(k, h)),
        "up": rng.normal(size=(k, h)),
        "down": rng.normal(size=(h, k)),
        "gamma": rng.uniform(0.2, 2, size=h),
        "epsilon": epsilon,
    }
    x, d, t = rng.normal(size=h), rng.normal(size=(4, h)), rng.normal(size=h)
    source = rng.normal(size=(3, h))
    actual = (
        apply_coefficients(
            coefficients(x, weights), rms_direction(x, d, weights), weights
        ).sum(0)
        @ t
    )
    np.testing.assert_allclose(
        d @ source_readout(x, t, weights), actual, rtol=1e-12, atol=1e-12
    )
    paired = alternatives(x, d, source, weights)["source_BN"].sum(0) @ t
    np.testing.assert_allclose(
        d @ mean_readout(source, t, weights), paired, rtol=1e-12, atol=1e-12
    )
    with pytest.raises(ValueError, match="independent"):
        mean_readout([], t, weights)
