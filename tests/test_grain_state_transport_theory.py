#!/usr/bin/env python3
"""Post-result mathematical witnesses, without changing the frozen experiment."""

import numpy as np

from grain_research.vector_mechanisms.source_state_transport import apply_map


def test_basis_nullspace_is_identity_for_both_maps():
    """A direction invisible to the input basis cannot acquire a residual update."""
    rng = np.random.default_rng(52)
    basis, _ = np.linalg.qr(rng.normal(size=(9, 3)))
    x, d = rng.normal(size=(11, 9)), rng.normal(size=(2, 11, 9))
    d -= (d @ basis) @ basis.T
    for mode, width in (("pointwise", 6), ("causal_context", 9)):
        model = {
            "mode": np.asarray(mode),
            "mean": np.zeros(9),
            "basis": basis,
            "center": np.zeros(width),
            "scale": np.ones(width),
            "coef": rng.normal(size=(width, 9)),
        }
        np.testing.assert_allclose(apply_map(x, d, model)[1], d, atol=1e-13)


def test_uniform_value_convergence_does_not_force_derivative_convergence():
    """sin(n*x)/n tends uniformly to zero but its derivative at zero stays one."""
    x = np.linspace(-1, 1, 101)
    for n in (10, 100, 1000):
        assert np.max(abs(np.sin(n * x) / n)) <= 1 / n
        assert np.cos(n * 0) == 1
