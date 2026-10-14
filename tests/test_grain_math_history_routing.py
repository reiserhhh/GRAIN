#!/usr/bin/env python3
"""Minimal falsifiable controls for route graph and observation composition."""

import numpy as np
import pytest

from grain_research.construction import math_history_routing as M


def graph(direct, relay):
    """Six tokens place two history tokens before a relay and current token."""
    return M.route_mask(6, [1, 2], [0, 3, 4], [5], [], direct, relay)


def test_route_support_and_relay_requires_two_layers():
    """Relay transmission is permitted only after an intermediate token update."""
    for d, r in ((0, 0), (1, 0), (0, 1), (1, 1)):
        a = graph(d, r)
        assert not np.triu(a, 1).any()
        assert bool(M.dependency_support(a, [1, 2], 1)[5]) == bool(d)
        assert bool(M.dependency_support(a, [1, 2], 2)[5]) == bool(d or r)
        assert bool(M.dependency_support(a, [1, 2], 17)[5]) == bool(d or r)


def test_padding_never_transmits():
    """Padding query states cannot reach other tokens when their keys are ignored."""
    a = M.route_mask(6, [1, 2], [0, 3, 4], [5], [2], 1, 1)
    assert not a[:, 2].any()
    assert not M.dependency_support(a, [2], 17)[5]


def test_invalid_partition_rejected():
    """Ambiguous support is not repaired from observed scores."""
    with pytest.raises(ValueError):
        M.route_mask(6, [1, 2], [0, 3, 5], [5], [], 1, 1)
    with pytest.raises(ValueError):
        M.route_mask(6, [1, 2], [0, 3, 4], [5], [0], 1, 1)


def test_mobius_signed_interaction_and_reconstruction():
    """A nonlinear route combination must retain positive and negative interaction."""
    f = np.array([[2, 3], [3, 5], [4, 4], [10, 1]], float)
    m = M.mobius(f)
    np.testing.assert_allclose(m.sum(0), f[3])
    np.testing.assert_allclose(m[3], [5, -5])


def test_author_laws_and_cycle_contrast():
    """Author laws hold margins and symmetric traffic fixed while changing order."""
    p, m = M.author_laws()
    assert np.min(p) > 0 and np.min(m) > 0
    np.testing.assert_allclose(p.sum(0), m.sum(0))
    np.testing.assert_allclose(p.sum(1), m.sum(1))
    np.testing.assert_allclose(p + p.T, m + m.T)
    f = np.arange(18).reshape(3, 3, 2) ** 2
    f[0, 1] += 3
    alt = (f[0, 1] + f[1, 2] + f[2, 0] - f[1, 0] - f[2, 1] - f[0, 2]) / 9
    np.testing.assert_allclose(M.author_contrast(f), alt)
    np.testing.assert_allclose(
        M.author_contrast(np.broadcast_to(f[0], f.shape)), 0, atol=1e-14
    )


def test_mobius_commutes_with_author_comparison():
    """Observed route composition and author averaging are the same linear account."""
    rng = np.random.default_rng(101)
    corners = rng.normal(size=(4, 3, 3, 2, 6))
    first = np.stack([M.author_contrast(f) for f in M.mobius(corners)])
    second = M.mobius(np.stack([M.author_contrast(f) for f in corners]))
    np.testing.assert_allclose(first, second, atol=1e-14)


def test_masks_removed_on_prefix_error(monkeypatch):
    """A failed forward cannot leave surgery hooks on the shared observer."""
    import types

    import torch

    from grain_research.vector_mechanisms import prefix

    def fail(*_args, **_kwargs):
        raise RuntimeError("test prefix failure")

    monkeypatch.setattr(prefix, "capture_prefix", fail)
    layers = [types.SimpleNamespace(self_attn=torch.nn.Identity()) for _ in range(17)]
    observer = types.SimpleNamespace(
        model=types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
    )
    row = {
        "input_ids": list(range(6)),
        "history": [1, 2],
        "relay": [0, 3, 4],
        "current": [5],
        "padding": [],
    }
    with pytest.raises(RuntimeError, match="test prefix failure"):
        M.capture_routed_prefix(observer, row, 0, 0)
    assert all(not layer.self_attn._forward_pre_hooks for layer in layers)
