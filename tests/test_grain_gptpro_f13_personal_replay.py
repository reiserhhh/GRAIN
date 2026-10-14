#!/usr/bin/env python3
"""Alternate precision recursion and explicit target covariances."""

import numpy as np

from grain_research.gptpro_foundation_v3.f13_dynamic_parameter import parameter_mixture
from scripts.audit_grain_gptpro_f13_personal_grid import reference_mixture


def test_alternate_replay_irregular_and_negative_dynamics():
    rng = np.random.default_rng(1310)
    parameters = rng.normal(size=(200, 3))
    times = np.array([0, 1, 3, 7, 8, 12])
    y, z = rng.normal(size=6), rng.normal(size=6)
    rows = [
        {"event_id": str(i), "t": int(t), "y": y[i], "z": z[i]}
        for i, t in enumerate(times)
    ]
    logs = np.log([0.5, 0.4])
    expected = parameter_mixture(parameters, rows, logs, [-1, 1, -1, 1])
    actual = reference_mixture(parameters, y, z, times, logs, [-1, 1, -1, 1])
    for k in expected:
        np.testing.assert_allclose(actual[k], expected[k], atol=1e-12)
