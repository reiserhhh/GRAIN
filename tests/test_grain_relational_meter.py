#!/usr/bin/env python3
"""Finite factorial observation maps, membership effects and role cross-talk."""

import numpy as np

from grain_research.relational_evidence import atom
from grain_research.relational_meter import (
    controlled_operator,
    enumerate_expectation,
    expected_joint,
    fractional_map,
)


def literal_controls():
    """Public signed world with balanced known role/channel opportunities."""
    values = np.zeros((4, 2, 2, 16))
    for g in range(4):
        for p in range(2):
            for sign in range(2):
                s = int(sign == p)
                values[g, p, sign] = atom(g // 2, g % 2, p, s).ravel()
    return values


def test_ideal_reader_recovers_selected_response_not_another_role():
    controls = literal_controls()
    offset, operator = controlled_operator(controls)
    mu = np.array([-0.2, 0.3, 0.5, -0.6])
    np.testing.assert_allclose(
        expected_joint(offset, operator, mu), enumerate_expectation(controls, mu)
    )
    value = fractional_map(offset, operator, actor=0, channel=1, at=mu)
    np.testing.assert_allclose(value["score"], 0.3)
    np.testing.assert_allclose(value["derivative"], [0, 1, 0, 0])


def test_fixed_soft_reader_has_affine_evidence_but_rational_conditional_score():
    rng = np.random.default_rng(301)
    confusion = rng.dirichlet(np.ones(16), size=16).T
    controls = np.einsum("ij,abkj->abki", confusion, literal_controls())
    offset, operator = controlled_operator(controls)
    a, b = rng.uniform(-0.4, 0.4, (2, 4))
    np.testing.assert_allclose(
        expected_joint(offset, operator, (a + b) / 2),
        (expected_joint(offset, operator, a) + expected_joint(offset, operator, b)) / 2,
    )
    result = fractional_map(offset, operator, at=a)
    for i in range(4):
        h = np.eye(4)[i] * 1e-5
        derivative = (
            fractional_map(offset, operator, at=a + h)["score"]
            - fractional_map(offset, operator, at=a - h)["score"]
        ) / 2e-5
        np.testing.assert_allclose(result["derivative"][i], derivative, atol=1e-9)
