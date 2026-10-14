#!/usr/bin/env python3
"""Exact observation tests, with null worlds that do not manufacture role effects."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_role_count_sectors as M
from grain_research.construction.math_heterogeneous_evidence_cube import basis, grid
from grain_research.vector_mechanisms.navigation import read_json


def test_orbit_projection_equals_count_conditioning_and_rank():
    identity = np.eye(32)
    p = M.count_projection(identity)
    np.testing.assert_allclose(p, M.permutation_projection(identity))
    np.testing.assert_allclose(p @ p, p)
    np.testing.assert_allclose(p.T, p)
    assert np.linalg.matrix_rank(p) == 16
    assert np.linalg.matrix_rank(np.eye(32) - p) == 16


def test_s3_keeps_count_and_role_is_orthogonal_not_deleted():
    a = np.random.default_rng(80).normal(size=(32, 3, 7))
    e = M.sectors(a)
    np.testing.assert_allclose(e["count"] + e["role"], a)
    assert abs(np.sum(e["count"] * e["role"])) < 1e-12
    np.testing.assert_allclose(M.count_projection(e["role"]), 0, atol=1e-15)
    np.testing.assert_allclose(e["pair_coefficients"], e["pair_count"] + e["pair_role"])


@pytest.mark.parametrize("power", (1, 2, 3, 5))
def test_arbitrary_nonlinear_count_null_has_no_role_pair(power):
    x, y, g, h, i = grid(5).T
    n = 2.5 + (g + h + i) / 2
    a = np.exp(x + y / 4) + x * y * n**power
    e = M.sectors(a)
    np.testing.assert_allclose(e["role"], 0, atol=1e-11)
    np.testing.assert_allclose(e["pair_role"], 0, atol=1e-11)


def test_role_pair_world_is_not_a_count_function():
    x, y, g, h, i = grid(5).T
    a = x * y * (g * h - g * i)
    e = M.sectors(a)
    np.testing.assert_allclose(e["count"], 0, atol=1e-15)
    np.testing.assert_allclose(e["pair_coefficients"], [1, -1, 0])
    np.testing.assert_allclose(e["pair_role"], [1, -1, 0])


@pytest.mark.parametrize("first,second", itertools.combinations(M.EDGES, 2))
def test_profiles_preserve_joint_content_count_and_lower_marginals(first, second):
    plus, minus = M.profiles(first, second)
    points = grid(5)
    assert min(plus.min(), minus.min()) > 0
    np.testing.assert_allclose([plus.sum(), minus.sum()], [1, 1])
    signs = basis(points, 5)
    masks = np.asarray(list(itertools.product((0, 1), repeat=5)))
    np.testing.assert_allclose((plus - minus) @ signs[:, masks.sum(1) <= 3], 0)
    for x, y, count in itertools.product((-1, 1), (-1, 1), range(4)):
        mask = (points[:, 0] == x) & (points[:, 1] == y)
        mask &= (points[:, 2:] > 0).sum(1) == count
        assert plus[mask].sum() == minus[mask].sum()
    a = np.random.default_rng(81).normal(size=(32, 4))
    e = M.sectors(a)
    k, l = M.EDGES.index(first), M.EDGES.index(second)
    np.testing.assert_allclose(
        plus @ a - minus @ a,
        0.5 * (e["pair_coefficients"][k] - e["pair_coefficients"][l]),
    )


def test_document_and_fixed_query_scope():
    root = Path(__file__).resolve().parents[1]
    material = read_json(root / "configs/grain_math_passive_event_composition_v1.json")
    docs = M.documents(material)
    assert len(docs) == len({r["body"] for r in docs}) == 1920
    assert len({(r["family"], r["focal"], r["companion"]) for r in docs}) == 60
    for row in docs:
        assert row["body"].count("Person A ") == row["person_a_events"]
        q = M.queries(row["focal"], row["companion"])
        assert len(set(q.values())) == 3
        assert q["focal"] == row["focal"] and q["companion"] == row["companion"]


@pytest.mark.parametrize("value", ([], np.zeros(16), np.full(32, np.nan)))
def test_incomplete_or_nonfinite_rejected(value):
    with pytest.raises(ValueError):
        M.sectors(value)
