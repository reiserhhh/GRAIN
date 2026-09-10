#!/usr/bin/env python3
"""Keep the frozen float32 baseline denominator when extending its range account."""

import numpy as np

from grain_research.vector_mechanisms import source_reader_motion as f
from tests.test_grain_source_reader_motion import model


def test_original_range_metric_float32_replay():
    m, rng = model()
    target = rng.normal(size=9).astype(np.float32)
    predicted = m["d"].mean(axis=0)
    a = f.base.range_account(m, target, predicted)
    b = f.range_account(m, target, predicted)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12)
    assert a["native_norm2"] == b["native_norm2"]
