#!/usr/bin/env python3
"""Finite support/bound tests without calibrating a preferred target outcome."""

import itertools

import numpy as np

from grain_research.construction import math_ternary_meaning as B
from grain_research.construction import math_ternary_trade as T


def test_safe_cube_step_keeps_lower_margins_and_increases_entropy():
    bits = np.array(list(itertools.product(range(2), repeat=4)))
    signs = (-1.0) ** bits.sum(1)
    values = np.random.default_rng(3).uniform(0.001, 2, (100, 16))
    tau, ell = T.safe_step(values, signs)
    for x, a, e in zip(values, tau, ell, strict=True):
        h = (x - a * signs).reshape(2, 2, 2, 2)
        f = x.reshape(2, 2, 2, 2)
        assert np.all(h > 0)
        for hf, ff in zip(B.margins(h), B.margins(f), strict=True):
            np.testing.assert_allclose(hf, ff, atol=1e-14)
        assert T.entropy(h) - T.entropy(f) + 1e-12 >= a * e / 2


def test_simultaneous_cube_overlap_bound_controls_a_full_joint_law():
    profiles = np.array(list(itertools.product(range(2), repeat=5)))
    f = np.random.default_rng(4).uniform(0.01, 2, (32, 8, 8, 8))
    f /= f.sum()
    h, changes, receipt = T.trade_reference(f, profiles)
    assert max(r["observed_max_incidence"] for r in receipt) == 135
    assert np.all(h >= f / 2)
    for x, y in zip(B.margins(h), B.margins(f), strict=True):
        np.testing.assert_allclose(x, y, atol=1e-15)
    np.testing.assert_allclose(h, f - changes.sum(0), atol=1e-16)
    assert T.entropy(h) - T.entropy(f) > 0


def test_target_components_cannot_change_the_other_four_target_meanings():
    profiles = np.array(list(itertools.product(range(2), repeat=5)))
    f = np.random.default_rng(5).uniform(0.01, 2, (32, 8, 8, 8))
    f /= f.sum()
    _, components, _ = T.trade_reference(f, profiles)
    for q in range(5):
        for other in range(5):
            if other != q:
                np.testing.assert_allclose(
                    components[q, profiles[:, other] == 1].sum(0), 0, atol=1e-17
                )


def test_zero_supported_cubes_are_retained_not_smoothed():
    profiles = np.array([[0] * 5, [1, 0, 0, 0, 0]])
    f = np.zeros((2, 8, 8, 8))
    f[:, 0, 0, 0] = [0.4, 0.6]
    h, change, receipt = T.trade_reference(f, profiles)
    np.testing.assert_array_equal(h, f)
    assert (
        not change.any()
        and sum(sum(r["supported_cubes_by_target"]) for r in receipt) == 0
    )


def test_source_bundle_reuses_only_the_strict_cold_schema_not_ipf():
    profiles = np.repeat([[0] * 5, [1, 0, 0, 0, 0]], 4, axis=0)
    p = np.random.default_rng(6).uniform(0.1, 2, (8, 2, 3, 8))
    p /= p.sum(-1, keepdims=True)
    b = T.build_bundle(p, profiles, tolerance=0, max_sweeps=0)
    B.validate_bundle(b)
    assert all(not r["ipf_used"] for rows in b["solver_receipts"] for r in rows)


def test_no_higher_order_target_assignment_produces_no_artificial_action():
    profiles = np.array(list(itertools.product(range(2), repeat=5)))
    rng = np.random.default_rng(7)
    g = rng.uniform(0.1, 2, 32)
    roles = rng.uniform(0.1, 2, (3, 8))
    f = np.einsum("g,a,b,c->gabc", g, *roles)
    f /= f.sum()
    h, changes, _ = T.trade_reference(f, profiles)
    np.testing.assert_allclose(h, f, atol=1e-16)
    np.testing.assert_allclose(changes, 0, atol=1e-16)


def test_target_triple_parity_is_detected_without_changing_any_lower_margin():
    profiles = np.array([[0] * 5, [1, 0, 0, 0, 0]])
    f = np.zeros((2, 8, 8, 8))
    for corner in itertools.product(range(2), repeat=4):
        f[corner] = (1 + 0.4 * (-1) ** sum(corner)) / 16
    h, changes, _ = T.trade_reference(f, profiles)
    for left, right in zip(B.margins(h), B.margins(f), strict=True):
        np.testing.assert_allclose(left, right, atol=1e-16)
    delta = B.fields(profiles, np.stack((f, h)))
    assert np.nanmax(np.abs(delta[0, 0] - delta[1, 0])) > 1e-5
    np.testing.assert_array_equal(changes[1:], 0)
