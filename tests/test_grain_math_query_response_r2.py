#!/usr/bin/env python3
"""The verifier must replay actual endpoints, not a reassociated float formula."""

import numpy as np

from grain_research.construction.math_position_support import physical_patches
from scripts.verify_grain_math_query_response_r2 import actual_step


def test_exact_realized_delta():
    rng = np.random.default_rng(3434)
    h = rng.normal(size=(4, 1536)).astype(np.float32)
    d = rng.normal(size=h.shape)
    step = actual_step(h, d, 0.01)
    delta = ((h + step).astype(float) - (h - step).astype(float)) / 2
    np.testing.assert_array_equal(delta, physical_patches(h, d, 0.01)["half_delta"])


def test_reassociated_step_changes_actual_measurement():
    rng = np.random.default_rng(3434)
    h = rng.normal(size=(4, 1536)).astype(np.float32)
    d = rng.normal(size=h.shape)
    original = actual_step(h, d, 0.01)
    reordered = (d * (0.01 * np.linalg.norm(h) / np.linalg.norm(d))).astype(np.float32)
    assert np.max(abs(original - reordered)) > 0
