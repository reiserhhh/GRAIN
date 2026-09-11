#!/usr/bin/env python3
"""Independent signed-error accounting for a composed local query contrast."""

import numpy as np

from grain_research.vector_mechanisms import teaching_composition as f


def test_composed_error_and_native_difference_oracle():
    rng = np.random.default_rng(861)
    p, p0 = rng.normal(size=(2, 8, 2, 4, 9))
    r0 = rng.normal(size=(8, 2, 9))
    m = rng.normal(size=9)
    d = rng.normal(size=(8, 9))
    r = np.stack((m + d / 2, m - d / 2), axis=1)
    a = f.decompose(p, p0, r, r0, m)
    e = f.energies(a)
    assert a["closure_max"] < 1e-12
    dn = r0[:, 0] - r0[:, 1]
    oracle = np.stack((m + dn / 2, m - dn / 2), axis=1)
    b = f.decompose(p, p0, oracle, r0, m)
    np.testing.assert_allclose(b["pieces"][:, 2], 0, atol=1e-12)
    np.testing.assert_allclose(
        np.sum((b["prediction"] - b["target"]) ** 2),
        e["oracle_native_difference_error2"],
    )
    good = f.decompose(p0, p0, oracle, r0, m)
    np.testing.assert_allclose(good["pieces"][:, 0], 0, atol=1e-12)
