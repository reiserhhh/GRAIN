#!/usr/bin/env python3
"""End-to-end finite examples for the frozen mathematical readout API."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.reference_system import (
    FrozenReferenceSystem,
    query_function_gram,
    reference_coordinates,
)


def fixture_system():
    """Make two non-identical query functions with fixed token identities."""
    rng = np.random.default_rng(61)
    arrays = {}
    for query in ("a", "b"):
        for key, shape in (("Z", (3, 4)), ("G", (3, 4)), ("h", (4,)), ("g", (4,))):
            arrays[f"{query}__{key}"] = rng.normal(size=shape)
        arrays[f"{query}__b"] = np.array(0.3)
    return FrozenReferenceSystem(
        arrays, {"a": [1, 2, 3], "b": [4, 5, 6]}, "observer", "reference"
    )


def test_complete_value_and_response_are_the_same_function():
    system = fixture_system()
    for q in ("a", "b"):
        a, tokens = system.arrays, system.tokens[q]
        z, h = a[f"{q}__Z"] + 0.7, a[f"{q}__h"] - 0.4
        result = system.response(q, z, h, z + 0.2, h - 0.1, tokens, "observer")
        expected = 0.3 + 0.7 * a[f"{q}__G"].sum() - 0.4 * a[f"{q}__g"].sum()
        assert result["full"]["score"] == pytest.approx(expected)
        assert result["natural"] == pytest.approx(result["edited"]["score"] - expected)
        assert result["anchor_candidate"] == pytest.approx(0.3 - expected)


def test_observer_tokens_and_nonfinite_states_are_not_silently_aligned():
    system = fixture_system()
    z, h = np.zeros((3, 4)), np.zeros(4)
    with pytest.raises(ValueError, match="Observer"):
        system.score("a", z, h, [1, 2, 3], "different")
    with pytest.raises(ValueError, match="token"):
        system.score("a", z, h, [3, 2, 1], "observer")
    with pytest.raises(ValueError, match="finite|Finite"):
        system.score("a", z * np.nan, h, [1, 2, 3], "observer")


def test_reference_position_does_not_depend_on_scoring_batch():
    values = np.array([[1.0, 2.0], [100.0, -40.0]])
    center, scale = [0.3, 0.4], [0.2, 0.6]
    np.testing.assert_allclose(
        reference_coordinates(values, center, scale)[0],
        reference_coordinates(values[0], center, scale),
    )
    with pytest.raises(ValueError, match="positive"):
        reference_coordinates(values, center, [0, 1])


def test_function_gram_preserves_shared_and_specific_coordinates():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 2))
    x = np.column_stack((x, x[:, 0] + x[:, 1]))
    gram = query_function_gram(x)
    np.testing.assert_allclose(gram, gram.T)
    assert np.linalg.eigvalsh(gram).min() >= -1e-12
    np.testing.assert_allclose(query_function_gram(x + [10, 20, -4]), gram, atol=1e-12)
