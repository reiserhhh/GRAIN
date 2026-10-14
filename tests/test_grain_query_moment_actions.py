#!/usr/bin/env python3
"""Finite-action identities, invariant stresses and BODY address tests."""

import numpy as np
import pytest

from grain_research.query_moment_actions import (
    DIRECTIONS,
    action_specs,
    array_hash,
    body_action,
    cell_reverse,
    odd_even,
    patch_body,
    quadratic_effect,
)
from grain_research.query_moment_operator import apply_operator, observation_moments


def fixture():
    """Repeated cells with a signed quadratic potential and nonzero affine arms."""
    rng = np.random.default_rng(13)
    g = np.repeat(rng.normal(size=(8, 4)), 5, axis=0)
    h = rng.normal(size=(len(g), 7))
    model = {
        "g_mean": g.mean(0),
        "g_scale": g.std(0),
        "query_basis": np.linalg.qr(rng.normal(size=(7, 3)))[0],
        "within_matrix": np.diag([0.4, -0.2, 0.1]),
        "beta1": rng.normal(size=(5, 7)),
        "beta2": rng.normal(size=(15, 7)),
        "offsets": np.array([0.1, -0.2, 0.3]),
    }
    return rng, g, h, model


def test_plan_counts_and_unique_ids():
    specs = action_specs()
    assert len(specs) == 31 and 16 * len(specs) == 496
    assert len({s["action"] for s in specs}) == len(specs)
    assert specs == action_specs()


@pytest.mark.parametrize("bad", [(), (0,), (-1,), (1,), (float("nan"),), (0.1, 0.1)])
def test_invalid_amplitudes(bad):
    with pytest.raises(ValueError):
        action_specs(bad)


def test_quadratic_finite_identity():
    rng, g, h, model = fixture()
    d, e = rng.normal(size=h.shape), 0.13
    y = apply_operator(g, h, model)[0]
    plus = apply_operator(g, h + e * d, model)[0]
    minus = apply_operator(g, h - e * d, model)[0]
    odd, even = odd_even(y, plus, minus)
    oq, eq = quadratic_effect(g, h, d, model, e)
    np.testing.assert_allclose(odd[2] - odd[1], oq, atol=1e-13)
    np.testing.assert_allclose(even, [0, 0, eq], atol=1e-13)


@pytest.mark.parametrize("action", ["cell_reverse", "cell_reflect"])
def test_invariants_exact_arithmetic(action):
    _, g, h, model = fixture()
    spec = next(s for s in action_specs() if s["action"] == action)
    changed = body_action(g, h, {}, spec)
    before = observation_moments(g, h, model)
    after = observation_moments(g, changed, model)
    for name in before:
        np.testing.assert_allclose(before[name], after[name], atol=1e-13)
    np.testing.assert_allclose(
        apply_operator(g, h, model)[0], apply_operator(g, changed, model)[0], atol=1e-13
    )
    assert np.linalg.norm(changed - h) > 0.1


def test_reversal_preserves_cell_multiset():
    _, g, h, _ = fixture()
    changed = cell_reverse(g, h)
    np.testing.assert_array_equal(cell_reverse(g, changed), h)
    for cell in np.unique(g, axis=0):
        positions = np.flatnonzero(np.all(g == cell, axis=1))
        np.testing.assert_array_equal(changed[positions], h[positions[::-1]])


def test_body_patch_is_quantized_and_has_no_context_effect():
    rng, g, h, _ = fixture()
    state = rng.normal(size=(len(h) + 3, h.shape[1])).astype(np.float32)
    pos = np.arange(1, len(h) + 1)
    original = state.copy()
    ds = {n: rng.normal(size=h.shape) for n in DIRECTIONS}
    changed = body_action(g, state[pos], ds, action_specs()[1])
    result = patch_body(state, pos, changed)
    assert changed.dtype == np.float32 and result.dtype == state.dtype
    np.testing.assert_array_equal(state, original)
    np.testing.assert_array_equal(result[pos], changed)
    np.testing.assert_array_equal(result[[0, -2, -1]], state[[0, -2, -1]])
    assert array_hash(changed) == array_hash(changed.copy())
    assert array_hash(changed) != array_hash(changed.astype(float))


@pytest.mark.parametrize(
    "pos",
    [
        np.array([1, 1]),
        np.array([2, 1]),
        np.array([1.0, 2.0]),
        np.array([-1, 2]),
        np.array([1, 4]),
    ],
)
def test_bad_addresses_rejected(pos):
    with pytest.raises(ValueError):
        patch_body(np.ones((5, 3), np.float32), pos, np.ones((2, 3), np.float32))
