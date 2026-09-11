#!/usr/bin/env python3
"""Finite support geometry and exact source observation-group contributions."""

import numpy as np

from grain_research.vector_mechanisms import coordinate_support as f


def test_far_input_returns_zero_coordinate_and_unit_coverage_remainder():
    rng = np.random.default_rng(8511)
    x, v = rng.normal(size=(2, 9, 3))
    g = np.repeat(np.arange(3), 3)
    m = f.base.fit(x, v, g > 0, rng.normal(size=(9, 5)), g, np.zeros(5), np.zeros(5))
    m["groups"] = g
    q = np.concatenate((x[:1], np.full((1, 3), 1e3)))
    a = f.account(m, q)
    assert a["coverage_remainder"][0] < 0.1
    np.testing.assert_allclose(a["coverage_remainder"][1], 1)
    np.testing.assert_allclose(a["parts"][1], 0)
    for i in range(2):
        np.testing.assert_allclose(
            a["parts"][i].sum(axis=0), f.base.field(m, q[i]), atol=1e-10
        )
