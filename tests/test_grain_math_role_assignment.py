#!/usr/bin/env python3
"""Synthetic tests for role-design identification, not personality validity."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_role_assignment as M
from grain_research.construction.math_heterogeneous_evidence_cube import (
    basis,
    coefficients,
    evaluate,
    grid,
)
from grain_research.construction.math_ownership_composition import (
    documents as old_documents,
)

ROOT = Path(__file__).resolve().parents[1]


def material():
    """Load the fixed public synthetic behaviour bank."""
    return json.loads(
        (ROOT / "configs/grain_math_passive_event_composition_v1.json").read_text()
    )


def test_complete_scope_and_old_reuse():
    """All target pairs and all old cells are preserved exactly."""
    old = {
        (r["family"], r["focal"], r["companion"], *r["coordinates"]): r
        for r in old_documents(material())
        if r["companion"] != "neutral"
    }
    new = M.documents(material())
    assert len(new) == 960
    assert len({(r["family"], r["focal"], r["companion"]) for r in new}) == 60
    reused = 0
    for row in new:
        x, y, g, h = row["coordinates"]
        if h == -g:
            assert (
                row["body"]
                == old[(row["family"], row["focal"], row["companion"], x, y, g)]["body"]
            )
            reused += 1
        assert row["person_a_events"] == row["body"].count("Person A")
    assert reused == 480
    assert (len(new) - reused) * len(M.TARGETS) == 2880


def test_single_role_flip_edits_only_its_event():
    """Independent assignments change the intended actor, not behaviour content."""
    rows = M.documents(material())
    lookup = {
        (r["family"], r["focal"], r["companion"], *r["coordinates"]): r for r in rows
    }
    for row in rows:
        for bit in (2, 3):
            point = row["coordinates"].copy()
            point[bit] *= -1
            partner = lookup[(row["family"], row["focal"], row["companion"], *point)]
            a, b = row["body"].splitlines(), partner["body"].splitlines()
            assert [i for i in range(3) if a[i] != b[i]] == [bit - 1]
            assert a[bit - 1].replace("Person A", "Person B") == b[bit - 1].replace(
                "Person A", "Person B"
            )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_all_restriction_aliases(seed):
    """Every character obeys the exact h=-g restriction formula."""
    c = np.random.default_rng(seed).normal(size=(16, 6, 3))
    values = evaluate(c, n=4)
    direct = coefficients(M.old_fraction(values), 3)
    assert np.allclose(direct, M.aliased_coefficients(c), atol=1e-13)
    assert np.allclose(M.effects(values)["old_gate"], 8 * (c[14] - c[13]), atol=1e-13)


def test_companion_and_anchor_worlds_are_indistinguishable_before_completion():
    """Two different mechanisms have exactly the same historical readings."""
    x, y, g, h = grid(4).T
    companion, anchor = x * y * g, -x * y * h
    assert np.array_equal(M.old_fraction(companion), M.old_fraction(anchor))
    assert not np.array_equal(companion, anchor)
    a, b = coefficients(companion, 4), coefficients(anchor, 4)
    assert a[14] == 1 and a[13] == 0
    assert b[14] == 0 and b[13] == -1


def test_four_way_signal_is_aliased_with_old_two_way_signal():
    """An old xy term can hide an independent joint actor modulation."""
    values = np.prod(grid(4), axis=1)
    old = coefficients(M.old_fraction(values), 3)
    assert old[6] == -1
    assert np.count_nonzero(old) == 1
    assert coefficients(values, 4)[15] == 1


def test_conditional_gates_from_direct_corners():
    """Main and context-dependent assignment effects remain distinct."""
    c = np.zeros((16, 2))
    c[12], c[13], c[14], c[15] = [0.2, 0.1], [-0.3, 0.4], [0.5, -0.2], [0.7, 0.9]
    e = M.effects(evaluate(c, n=4))
    for gi, g in enumerate((-1, 1)):
        for hi, h in enumerate((-1, 1)):
            assert np.allclose(
                e["mixed"][gi, hi], 4 * (c[12] + g * c[14] + h * c[13] + g * h * c[15])
            )
        assert np.allclose(e["anchor_gate_at_g"][gi], 8 * (c[13] + g * c[15]))
    for hi, h in enumerate((-1, 1)):
        assert np.allclose(e["companion_gate_at_h"][hi], 8 * (c[14] + h * c[15]))


def test_profile_pair_preserves_all_lower_order_marginals():
    """Joint actor/content observations can matter when lower marginals agree."""
    plus, minus = M.profile_pair()
    assert plus.sum() == minus.sum() == 1
    assert np.all(plus > 0) and np.all(minus > 0)
    for order in range(4):
        for selected in itertools.combinations(range(4), order):
            character = np.prod(grid(4)[:, selected], axis=1)
            assert plus @ character == minus @ character
    c = np.random.default_rng(4).normal(size=(16, 6))
    values = evaluate(c, n=4)
    assert np.allclose(plus @ values - minus @ values, c[15], atol=1e-13)


def test_full_basis_is_identifiable_not_rank_selected():
    """The full design has sixteen orthogonal columns without fitting."""
    b = basis(grid(4), 4)
    assert np.array_equal(b.T @ b, 16 * np.eye(16))
    old = M.old_fraction(b)
    assert np.linalg.matrix_rank(old) == 8


@pytest.mark.parametrize("values", [np.zeros(8), np.full(16, np.nan), np.asarray(0.0)])
def test_invalid_complete_tables_fail(values):
    """Missing or nonfinite observations cannot be replaced by passing constants."""
    with pytest.raises(ValueError):
        M.complete(values)


@pytest.mark.parametrize("epsilon", [0, -0.5, 1.1, np.nan])
def test_invalid_compositions_fail(epsilon):
    """Composition witnesses must be valid probabilities with a nonzero contrast."""
    with pytest.raises(ValueError):
        M.profile_pair(epsilon)
