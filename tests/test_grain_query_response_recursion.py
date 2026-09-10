#!/usr/bin/env python3
"""Check the exact recurrence without assuming common, invertible or commuting maps."""

import numpy as np

from grain_research.vector_mechanisms.query_response_recursion import (
    symmetric_step,
    terminal_parts,
)


def test_noncommuting_singular_response_recursion():
    rng = np.random.default_rng(46)
    a = rng.normal(size=(2, 5, 4, 4))
    a[0, 2, 0] = 0
    d = rng.normal(size=(2, 3, 4))
    r = terminal_parts(a, d)
    np.testing.assert_allclose(
        r["terminal_parts"].sum(0), r["difference"], rtol=1e-12, atol=1e-12
    )
    assert r["terminal_parts"].shape == (6, 3, 4)
    c, i = symmetric_step(a[0, 0], a[1, 0], d[0], d[1])
    np.testing.assert_allclose(c + i, d[0] @ a[0, 0].T - d[1] @ a[1, 0].T)


def test_same_input_does_not_mean_same_response_but_common_operators_add_no_injection():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(2, 3, 4, 4))
    d = rng.normal(size=(2, 2, 4))
    d[1] = d[0]
    r = terminal_parts(a, d)
    np.testing.assert_array_equal(r["terminal_parts"][0], 0)
    assert np.linalg.norm(r["difference"]) > 0
    a[1] = a[0]
    d[1] = 0
    r = terminal_parts(a, d)
    np.testing.assert_array_equal(r["local_injections"], 0)
    np.testing.assert_allclose(r["terminal_parts"][0], r["difference"])
