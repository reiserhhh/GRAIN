#!/usr/bin/env python3
"""Exact factorial, conditional projection, and non-unique allocation witnesses."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.conditional_factorial import (
    ORDERS,
    factorial_parts,
    ordered_parts,
)
from grain_research.vector_mechanisms.projected_components import projected_components


def test_additive_and_pure_interaction_cells():
    e = np.arange(1.0, 7.0)
    x = e[::-1]
    zero = np.zeros(len(e))
    additive = factorial_parts(np.column_stack([e + x, x, e, zero]))
    np.testing.assert_allclose(additive, np.column_stack([zero, e, x, zero]))
    pure = factorial_parts(np.column_stack([e * x, zero, zero, zero]))
    np.testing.assert_allclose(pure[:, -1], e * x)
    np.testing.assert_allclose(
        ordered_parts(pure, "symmetric")[:, 1:], np.column_stack([e * x / 2, e * x / 2])
    )


def test_interaction_moves_between_orders_without_double_counting():
    parts = np.random.default_rng(33).normal(size=(25, 4))
    ex = ordered_parts(parts, ORDERS[0])
    xe = ordered_parts(parts, ORDERS[1])
    sym = ordered_parts(parts, ORDERS[2])
    for allocated in (ex, xe, sym):
        np.testing.assert_allclose(allocated.sum(1), parts.sum(1), atol=1e-14)
    np.testing.assert_allclose(xe[:, 1] - ex[:, 1], parts[:, 3], atol=1e-14)
    np.testing.assert_allclose(xe[:, 2] - ex[:, 2], -parts[:, 3], atol=1e-14)
    np.testing.assert_allclose(sym, (ex + xe) / 2, atol=1e-14)


def test_projector_commutes_with_declared_allocations_and_old_anchor_split():
    rng = np.random.default_rng(23)
    parts, p, y = rng.normal(size=(32, 4)), rng.normal(size=32), rng.normal(size=32)
    total = parts.sum(1)
    original = np.asarray(projected_components(total, parts, p, y)["contributions"])
    for order, w in zip(ORDERS, (0.0, 1.0, 0.5)):
        actual = projected_components(total, ordered_parts(parts, order), p, y)[
            "contributions"
        ]
        expected = [
            original[0],
            original[1],
            original[2] + w * original[4],
            original[3] + (1 - w) * original[4],
        ]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    merged = np.column_stack([parts[:, 0] + parts[:, 1], parts[:, 2], parts[:, 3]])
    account = projected_components(total, merged, p, y)["contributions"]
    np.testing.assert_allclose(account[1], original[1] + original[2], atol=1e-14)


def test_marginal_criterion_signal_can_already_be_in_the_proxy():
    rng = np.random.default_rng(9)
    z, _ = np.linalg.qr(rng.normal(size=(40, 2)) - 0.3)
    p, r = z.T
    parts = np.column_stack([p, r, -r, np.zeros(len(r))])
    account = projected_components(p, parts, p, p)
    assert np.corrcoef(parts[:, 0], p)[0, 1] > 0.999
    np.testing.assert_allclose(account["contributions"], [1, 0, 0, 0, 0], atol=1e-14)


def test_bad_cells_or_undeclared_order_fail():
    with pytest.raises(ValueError):
        factorial_parts(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        factorial_parts(np.full((4, 4), np.nan))
    with pytest.raises(ValueError):
        ordered_parts(np.zeros((4, 4)), "best_y")
