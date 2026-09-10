#!/usr/bin/env python3
"""Post-account check of the infinitesimal selection/direction decomposition."""

import numpy as np

from grain_research.vector_mechanisms.query_partition_geometry import partition


def test_query_covariance_derivative_is_direction_plus_third_cumulant():
    """Differentiate fixed K/value/dK evidence; no empirical target or new fit."""
    rng = np.random.default_rng(730)
    k, dk = rng.normal(size=(13, 4)), rng.normal(size=(13, 4))
    q, u, v = rng.normal(size=4), rng.normal(size=4), rng.normal(size=13)
    l, d, b, dd = k @ q, dk @ q, k @ u, dk @ u
    p = np.exp(l - l.max())
    p /= p.sum()
    centered = lambda z: z - p @ z
    orientation = (p * centered(v)) @ centered(dd)
    selection = np.sum(p * centered(v) * centered(d) * centered(b))

    def response(theta):
        z = q + theta * u
        return partition(k @ z, np.column_stack((v, dk @ z)))["covariance"][0, 1]

    fd = (response(1e-5) - response(-1e-5)) / 2e-5
    np.testing.assert_allclose(fd, orientation + selection, atol=1e-8)
