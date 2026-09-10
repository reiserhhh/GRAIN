#!/usr/bin/env python3
"""Crossed scalar paths, covariance operators and finite swap identities."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import query_selection_orientation as f
from tests.test_grain_query_partition_geometry import toy


@pytest.mark.parametrize("seed", [73, 74])
def test_all_crossed_scalar_derivatives_and_operator(seed):
    q, e = toy(seed)
    q = q[:2]
    result = f.crossed(q, e)
    for a in range(2):
        for b in range(2):
            for factor in range(4):
                fd = (
                    f.path_score(q, e, a, b, factor, 1e-5)
                    - f.path_score(q, e, a, b, factor, -1e-5)
                ) / 2e-5
                np.testing.assert_allclose(
                    fd, result["actions"][a, b, :, factor], atol=1e-8
                )
    # Once the selection operator is fixed, direction-anchor interpolation is linear.
    mix = result["anchors"].mean(0, keepdims=True)
    np.testing.assert_allclose(
        f.operator_actions(result["operator"], mix)[:, 0],
        result["actions"].mean(1),
        atol=1e-12,
    )


def test_identical_anchors_zero_swap_and_constant_value_zero_operator():
    q, e = toy()
    q = q[:2]
    q[1] = q[0]
    a = f.crossed(q, e)["actions"]
    for z in f.decompose(a).values():
        np.testing.assert_array_equal(z, 0)
    e["v"][:] = 0
    np.testing.assert_array_equal(f.crossed(q, e)["operator"], 0)


def test_pure_selection_direction_and_symmetric_interaction_allocation():
    y = np.arange(1.0, 6.0)
    a = np.stack([np.stack((y, y)), np.stack((3 * y, 3 * y))])
    np.testing.assert_array_equal(f.decompose(a)["selection"], 2 * y)
    np.testing.assert_array_equal(f.decompose(a)["direction"], 0)
    b = a.swapaxes(0, 1)
    np.testing.assert_array_equal(f.decompose(b)["direction"], 2 * y)
    a[1, 1] += 4 * y
    d = f.decompose(a)
    np.testing.assert_array_equal(d["interaction"], 4 * y)
    np.testing.assert_array_equal(d["selection"], 4 * y)
    np.testing.assert_array_equal(d["direction"], 2 * y)
    r = f.account(a, y)
    assert r["selection_swap_projection"] + r[
        "direction_swap_projection"
    ] == pytest.approx(1)
    assert r["selection_energy_over_swap"] + r["direction_energy_over_swap"] + r[
        "cross_energy_over_swap"
    ] == pytest.approx(1)


def test_undeclared_current_q_and_shape_errors_rejected():
    q, e = toy()
    with pytest.raises(ValueError, match="Exactly two"):
        f.crossed(q, e)
    with pytest.raises(ValueError, match="Only pure"):
        f.crossed(q[:2], e | {"q": q})
    with pytest.raises(ValueError, match="two-by-two"):
        f.decompose(np.ones((3, 2)))
