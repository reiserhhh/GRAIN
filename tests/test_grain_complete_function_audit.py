#!/usr/bin/env python3
"""Regression test for the independent audit's source-target precision contract."""

import numpy as np

from grain_research.vector_mechanisms.complete_function_extraction import fit_scalar
from tests.test_grain_complete_function_extraction import sample


def test_source_target_reductions_promote_float32():
    """Native FP32 targets use the estimator's float64 source normalization."""
    x, d, _, t = sample()
    y = np.asarray([0.1234567, 6.34567, 1.23, 0.34], dtype=np.float32)
    model, _ = fit_scalar(x, d, y, t, "joint", range(4))
    assert float(model["origin"]) == y.astype(float).mean()
    assert float(model["target_scale"]) == y.astype(float).std()
