#!/usr/bin/env python3
"""Dual contraction, observable-null and physical error-bound witnesses."""

import numpy as np
import pytest

from grain_research.construction import math_route_action_audit as A
from grain_research.construction import math_text_query_route_cut as R


def test_dual_and_primal_routes_match():
    rng = np.random.default_rng(4202)
    j, d, c, p, e = (
        rng.normal(size=s) for s in ((3, 3, 2, 4), (2, 2, 4), (3, 3), (3, 5, 4), (5, 4))
    )
    v = R.link_routes(j, d, c, p)
    np.testing.assert_allclose(
        A.dual_route_actions(j, d, c, p, e), np.einsum("gapd,pd->ga", v, e), atol=1e-12
    )
    witness = A.action_error_witness(j.sum(0), d, c, p, e)
    np.testing.assert_allclose(
        witness["signed_error"], A.dual_route_actions(j, d, c, p, e).sum(0), atol=1e-12
    )
    assert np.all(abs(witness["signed_error"]) <= witness["directional_bound"] + 1e-12)
    assert np.all(witness["directional_bound"] <= witness["frobenius_bound"] + 1e-12)


def test_coordinate_error_can_be_physically_null():
    delta = np.array([[[0.0, 4.0]], [[0.0, 8.0]]])
    d = np.array([[[1.0, 0.0]], [[2.0, 0.0]]])
    witness = A.action_error_witness(
        delta, d, np.eye(2), np.eye(2).reshape(2, 1, 2), np.ones((1, 2))
    )
    np.testing.assert_array_equal(witness["signed_error"], 0)
    assert np.max(abs(delta)) == 8 and witness["frobenius_bound"][0] > 0


def test_bound_is_not_invariant_to_direction_units():
    delta = np.ones((2, 1, 2))
    d = np.ones((2, 1, 2))
    c, p, e = np.eye(2), np.eye(2).reshape(2, 1, 2), np.ones((1, 2))
    old = A.action_error_witness(delta, d, c, p, e)
    new = A.action_error_witness(delta, d * 7, c, p, e * 3)
    np.testing.assert_allclose(new["signed_error"], old["signed_error"] * 21)
    np.testing.assert_allclose(new["directional_bound"], old["directional_bound"] * 21)


def test_nonfinite_bound_rejected():
    with pytest.raises(ValueError):
        A.action_error_witness(
            np.full((2, 1, 2), np.nan),
            np.ones((2, 1, 2)),
            np.eye(2),
            np.eye(2).reshape(2, 1, 2),
            np.ones((1, 2)),
        )
