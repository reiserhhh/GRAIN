#!/usr/bin/env python3
"""Vectorized source likelihood versus independent dense Gaussian likelihood."""

import numpy as np

from grain_research.gptpro_foundation_v3.f05_recovery import dense_posterior
from scripts.audit_grain_gptpro_f13_dynamic_sources import batch_nll


def test_source_likelihood_dense_reference():
    rng = np.random.default_rng(1339)
    psi = rng.normal(size=(10, 3))
    y = rng.normal(size=(10, 16))
    z = rng.choice([-1.0, 1.0], size=y.shape)
    logs = np.log([0.5, 0.4])
    expected = 0.0
    for i in range(10):
        rows = [
            {"event_id": str(t), "t": t, "y": y[i, t], "z": z[i, t]} for t in range(16)
        ]
        expected -= dense_posterior(psi[i], rows, logs)["loglik"]
    np.testing.assert_allclose(batch_nll(psi, y, z, logs), expected, atol=1e-10)
