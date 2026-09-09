#!/usr/bin/env python3
"""Check residual reachability and exact A-to-B contraction on synthetic states."""

import numpy as np

from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.state_geometry import (
    compile_projection,
    decompose_state_error,
    residual_basis,
    score_compiled_projection,
)
from grain_research.vector_mechanisms.state_transport import (
    fit_transport,
    reconstruct,
    unpack_state,
)


def test_reachability_is_orthogonal_and_is_not_an_independent_prediction():
    rng = np.random.default_rng(120)
    x = rng.normal(size=(8, 12))
    y = x + rng.normal(size=(8, 3)) @ rng.normal(size=(3, 12))
    model = fit_transport(x, y)
    basis = residual_basis(model)
    early, actual = rng.normal(size=(2, 12))
    estimate = reconstruct(model, early)["transport1"]
    oracle, outside, inside, metrics = decompose_state_error(
        model, basis, early, actual, estimate
    )
    assert len(basis) == 3
    np.testing.assert_allclose(actual - estimate, outside + inside, atol=1e-12)
    assert metrics["orthogonality_accounting_error"] < 1e-10
    assert abs(outside @ inside) < 1e-10
    assert np.linalg.norm(actual - oracle) <= np.linalg.norm(actual - estimate) + 1e-12


def test_compiled_projections_equal_generated_state_reader_for_both_kernels():
    rng = np.random.default_rng(121)
    x, y = rng.normal(size=(2, 30, 12))
    model = fit_transport(x, y)
    f = FunctionalQuery(
        rng.normal(size=(2, 4)),
        rng.normal(size=4),
        rng.normal(size=(2, 4)),
        rng.normal(size=4),
        0.3,
    )
    compiled = compile_projection(model, f)
    for _ in range(6):
        early = rng.normal(size=12)
        states = reconstruct(model, early)
        for degree in (1, 2):
            actual = f.parts(*unpack_state(states[f"transport{degree}"], (2, 4)))
            predicted = score_compiled_projection(compiled, early, degree)
            np.testing.assert_allclose(
                list(actual.values()), list(predicted.values()), atol=1e-11
            )
    assert compiled["coef1"].shape == (30, 3)
