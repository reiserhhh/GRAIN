#!/usr/bin/env python3
"""Algebra, exact source intervention, and exception cleanup for S24."""

import numpy as np
import pytest
import torch

from grain_research.vector_mechanisms.configuration_mediation import (
    endpoint_errors,
    factorial_allocation,
    projected_body,
)


def test_two_factor_additive_reconstruction():
    rng = np.random.default_rng(24)
    cells = rng.normal(size=(30, 4))
    result = factorial_allocation(cells)
    np.testing.assert_allclose(result["additive_error"], 0, atol=1e-15)
    np.testing.assert_allclose(
        result["head_native_body"] - result["head_projected_body"],
        result["interaction"],
        atol=1e-15,
    )


def test_pure_interaction_is_split_equally_not_discarded():
    result = factorial_allocation([8, 0, 0, 0])
    assert result["interaction"] == 8
    assert result["head_allocation"] == result["other_allocation"] == 4
    reverse = factorial_allocation([0, 8, 0, 0])
    assert reverse["head_allocation"] == reverse["other_allocation"] == -4


def test_frozen_endpoints_and_invalid_cells():
    assert endpoint_errors([1, 2, 3, 4], 1, 2) == {"native": 0, "projected": 0}
    with pytest.raises(ValueError, match="replay"):
        endpoint_errors([1, 2, 3, 4], 1, 3)
    with pytest.raises(ValueError, match="Finite"):
        factorial_allocation([1, 2, float("nan"), 4])


@pytest.mark.parametrize("tuple_output", [False, True])
def test_projection_preserves_query_and_means_and_cleans_up(tuple_output):
    class Layer(torch.nn.Module):
        def forward(self, value):
            return (value, "cache") if tuple_output else value

    layer = Layer()
    original = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    groups = [np.array([0, 1]), np.array([3, 4])]
    with pytest.raises(RuntimeError, match="injected"), projected_body(layer, groups):
        result = layer(original)
        state = result[0] if tuple_output else result
        for group in groups:
            expected = original[0, group].mean(0).expand(2, -1)
            torch.testing.assert_close(state[0, group], expected)
        torch.testing.assert_close(state[:, [2, 5]], original[:, [2, 5]])
        if tuple_output:
            assert result[1] == "cache"
        raise RuntimeError("injected")
    assert not layer._forward_hooks
    result = layer(original)
    torch.testing.assert_close(result[0] if tuple_output else result, original)
