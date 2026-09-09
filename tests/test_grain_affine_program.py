#!/usr/bin/env python3
"""Affine compilation is an algebraic identity, including nonzero kernel centering."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.affine_program import compile_affine, score_affine
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.source_span import fit_linear, predict_linear
from grain_research.vector_mechanisms.state_transport import unpack_state


def test_affine_compilation_and_constant_derivative():
    rng = np.random.default_rng(11)
    shape = (4, 5)
    dim = (shape[0] + 1) * shape[1]
    x, y = rng.normal(size=(2, 9, dim))
    model = fit_linear(x, y)
    f = FunctionalQuery(
        rng.normal(size=shape),
        rng.normal(size=5),
        rng.normal(size=shape),
        rng.normal(size=5),
        0.3,
    )
    compiled = compile_affine(model, f)
    assert set(compiled) == {"coefficients", "offset", "anchor"}
    a, b = rng.normal(size=(2, dim))
    for value in (a, b):
        exact = f.parts(*unpack_state(predict_linear(model, value), shape))
        got = score_affine(compiled, value)
        for key in exact:
            assert got[key] == pytest.approx(exact[key], abs=1e-12)
    delta = score_affine(compiled, b)["score"] - score_affine(compiled, a)["score"]
    assert delta == pytest.approx(
        float((b - a) @ compiled["coefficients"].sum(1)), abs=1e-12
    )
    with pytest.raises(ValueError):
        score_affine(compiled, np.ones(3))


def test_compilation_does_not_assume_exact_zero_centering():
    rng = np.random.default_rng(14)
    x, y = rng.normal(size=(2, 6, 9))
    model = fit_linear(x, y)
    # Check the general formula when the persisted normalized anchors have a shift.
    model["train_x"] += 0.2
    model["col1"] += np.arange(6) * 0.1
    model["grand1"] += 0.3
    f = FunctionalQuery(np.ones((2, 3)), np.ones(3), np.ones((2, 3)), np.ones(3), 0.0)
    compiled = compile_affine(model, f)
    z = rng.normal(size=9)
    exact = f.parts(*unpack_state(predict_linear(model, z), (2, 3)))
    assert score_affine(compiled, z)["score"] == pytest.approx(
        exact["score"], abs=1e-12
    )
