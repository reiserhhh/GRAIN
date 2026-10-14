"""Regression tests for the pre-HM17 context operator's original public API."""

import numpy as np
import pytest

from grain_research.construction import math_context_operator as C


def model():
    """Nonzero-sum dual makes omitted row-centering visible."""
    rng = np.random.default_rng(19)
    return {
        "kind": "tensor",
        "a": rng.normal(size=(7, 3)),
        "b": rng.normal(size=(7, 4)),
        "dual": rng.normal(size=7),
        "column_mean": rng.normal(size=7),
        "grand_mean": 0.31,
        "bias": -0.17,
    }


def test_original_compiled_score_retains_row_centering():
    m, rng = model(), np.random.default_rng(20)
    a, o = rng.normal(size=(5, 3)), rng.normal(size=(5, 4))
    kernel = (1 + a @ m["a"].T) * (1 + o @ m["b"].T)
    centered = (
        kernel - kernel.mean(axis=1)[:, None] - m["column_mean"] + m["grand_mean"]
    )
    expected = centered @ m["dual"] + m["bias"]
    np.testing.assert_allclose(
        C.score_compiled(C.compile_tensor(m), a, o), expected, atol=1e-12
    )


def test_original_context_response_is_exact():
    op, rng = C.compile_tensor(model()), np.random.default_rng(21)
    a, o, v = rng.normal(size=(5, 3)), rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    additive, coupled, total = C.context_response(op, a, o, v)
    np.testing.assert_allclose(additive + coupled, total, atol=1e-12)
    np.testing.assert_allclose(
        total, C.score_compiled(op, a, v) - C.score_compiled(op, a, o), atol=1e-12
    )


def test_original_four_corner_is_exact():
    op, rng = C.compile_tensor(model()), np.random.default_rng(22)
    a, b, o, v = (
        rng.normal(size=(5, 3)),
        rng.normal(size=(5, 3)),
        rng.normal(size=(5, 4)),
        rng.normal(size=(5, 4)),
    )
    expected = (
        C.score_compiled(op, b, v)
        - C.score_compiled(op, a, v)
        - C.score_compiled(op, b, o)
        + C.score_compiled(op, a, o)
    )
    np.testing.assert_allclose(C.four_corner(op, a, b, o, v), expected, atol=1e-12)


def test_original_nontensor_model_is_rejected():
    m = {**model(), "kind": "linear"}
    with pytest.raises(ValueError):
        C.compile_tensor(m)


def test_original_misaligned_dual_is_rejected():
    m = {**model(), "dual": np.ones(6)}
    with pytest.raises(ValueError):
        C.compile_tensor(m)


def test_original_misaligned_context_is_rejected():
    op = C.compile_tensor(model())
    with pytest.raises(ValueError):
        C.context_response(op, np.ones((2, 3)), np.ones((2, 4)), np.ones((3, 4)))
