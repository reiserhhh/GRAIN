"""Synthetic source-only projection and native factorial tests."""

import numpy as np
import pytest

from grain_research.construction import math_grain_native_remainder as M


def rows():
    """Return the declared96 source corners in deterministic source order."""
    return [
        dict(variant=0, wording=w, family=f, condition=c, polarity=b, owner=o)
        for w in range(2)
        for f in range(6)
        for c in (-1, 1)
        for b in (-1, 1)
        for o in (-1, 1)
    ]


def test_orthogonal_remainder_and_action_account():
    rng = np.random.default_rng(124)
    g, x = rng.normal(size=(96, 3, 4)), rng.normal(size=(96, 30))
    result = M.source_decomposition(g, x, rows())
    np.testing.assert_allclose(
        result["design"].T @ result["remainder_values"], 0, atol=1e-12
    )
    np.testing.assert_allclose(
        result["represented"] + result["remainder_values"], x, atol=1e-12
    )
    np.testing.assert_allclose(
        result["design"] @ result["coefficient"], result["represented"], atol=1e-12
    )
    np.testing.assert_allclose(
        result["grain_part"] + result["remainder"], result["whole"], atol=1e-12
    )


def test_represented_values_have_zero_remainder():
    rng = np.random.default_rng(125)
    g = rng.normal(size=(96, 3, 4))
    x = np.column_stack((np.ones(96), g.reshape(96, 12))) @ rng.normal(size=(13, 30))
    result = M.source_decomposition(g, x, rows())
    np.testing.assert_allclose(result["remainder_values"], 0, atol=1e-12)


def test_affine_coordinate_change_and_row_order_do_not_change_projector():
    rng = np.random.default_rng(126)
    g, x = rng.normal(size=(96, 3, 4)), rng.normal(size=(96, 30))
    a = M.source_decomposition(g, x, rows())
    transformed = (
        g.reshape(96, 12) @ (np.eye(12) + 0.1 * rng.normal(size=(12, 12))) + 2
    ).reshape(g.shape)
    b = M.source_decomposition(transformed, x, rows())
    np.testing.assert_allclose(a["projector"], b["projector"], atol=1e-12)
    order = rng.permutation(96)
    c = M.source_decomposition(g[order], x[order], [rows()[i] for i in order])
    np.testing.assert_allclose(c["whole"], a["whole"], atol=1e-12)
    np.testing.assert_allclose(c["remainder"], a["remainder"], atol=1e-12)


def test_source_boundary_and_missing_cell_rejected():
    g, x = np.ones((96, 3, 4)), np.ones((96, 30))
    bad = rows()
    bad[0] = dict(bad[0], variant=1)
    with pytest.raises(ValueError):
        M.source_decomposition(g, x, bad)
    bad = rows()
    bad[0] = dict(bad[1])
    with pytest.raises(ValueError):
        M.source_decomposition(g, x, bad)


def test_random_control_is_head_norm_matched():
    ref = np.random.default_rng(127).normal(size=(6, 2, 18))
    actual = M.random_control(ref, (3, 2, 3))
    np.testing.assert_allclose(
        np.linalg.norm(actual, axis=(2, 4)),
        np.linalg.norm(ref.reshape(6, 2, 3, 2, 3), axis=(2, 4)),
    )


def test_behavior_owner_focus_and_nonlinear_account():
    n = np.zeros((5, 2, 6, 2, 2, 2, 6, 4))
    amplitudes = [10, 8, 7, 4, 10]
    for arm, b, o, focus in np.ndindex(5, 2, 2, 2):
        n[arm, :, :, b, o, focus] = (
            amplitudes[arm] * (2 * b - 1) * (2 * o - 1) * (1 - 2 * focus)
        )
    f = M.actor_focus_field(n)
    np.testing.assert_allclose(f["baseline"], 10)
    np.testing.assert_allclose(f["interaction"], 1)
    assert len(M.response_summary(f)) == 16
