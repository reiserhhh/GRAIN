#!/usr/bin/env python3
"""Distinct mechanisms, missing opportunity and complete signed accounting."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_choice_geometry as M
from grain_research.construction import math_condition_shape as S
from scripts.run_grain_choice_geometry_hm188 import literal_fields, literal_observations


def test_balanced_condition_keeps_geometry_without_D():
    x = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [1.0, 2.0]])
    m = M.observations(x, np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(m["geometry"], [[1, 2], [2, 4]])
    np.testing.assert_array_equal(m["weighted"], 0)
    assert m["between"].sum() > 0


@pytest.mark.parametrize("value", [0, 1])
def test_missing_geometry_not_neutral(value):
    m = M.observations(np.arange(8).reshape(4, 2), np.full(4, value))
    assert np.isnan(m["geometry"]).all()
    np.testing.assert_array_equal(m["weighted"], 0)
    with pytest.raises(ValueError, match="No supported"):
        M.reference(np.zeros(2), np.full((2, 2, 2), np.nan), np.zeros(2, bool))


def test_same_marginals_distinct_author_coupling():
    a = np.array([0.08, -0.08])
    g = np.array([np.diag([4.0, 0.0]), np.diag([0.0, 4.0])])
    s = np.ones(2, bool)
    first, swapped = M.reference(a, g, s), M.reference(a[::-1], g, s)
    for k in ("A", "G", "q", "product"):
        np.testing.assert_allclose(first[k], swapped[k])
    np.testing.assert_allclose(first["L"], -swapped["L"])
    parts = M.contrast(swapped, first)
    np.testing.assert_array_equal(parts[:3], 0)
    assert np.linalg.norm(parts[3]) > 0
    np.testing.assert_allclose(parts[:4].sum(0), parts[4])


def test_coverage_only_is_not_geometry_imputation():
    g = np.array(
        [np.eye(2), np.eye(2), np.full((2, 2), np.nan), np.full((2, 2), np.nan)]
    )
    first = M.reference(np.array([0.08, 0.08, 0, 0]), g, np.array([1, 1, 0, 0], bool))
    second = M.reference(np.array([0.08, 0.08]), g[:2], np.ones(2, bool))
    parts = M.contrast(first, second)
    np.testing.assert_allclose(parts[0], parts[4])
    np.testing.assert_array_equal(parts[1:4], 0)


def test_choice_only_geometry_only_and_orthogonal_frame():
    s = np.ones(4, bool)
    g = np.tile(np.diag([2.0, 1.0]), (4, 1, 1))
    first = M.reference(np.full(4, 0.04), g, s)
    choice = M.contrast(first, M.reference(np.full(4, 0.08), g, s))
    np.testing.assert_allclose(choice[1], choice[4])
    geometry = M.contrast(first, M.reference(np.full(4, 0.04), 2 * g, s))
    np.testing.assert_allclose(geometry[2], geometry[4])
    q, _ = np.linalg.qr(np.array([[1.0, 2.0], [3.0, 4.0]]))
    rotated = M.contrast(
        M.reference(np.full(4, 0.04), q.T @ g @ q, s),
        M.reference(np.full(4, 0.04), q.T @ (2 * g) @ q, s),
    )
    np.testing.assert_allclose(rotated, q.T @ geometry @ q, atol=1e-14)


def sample():
    """Small positive-support world with two target groups in both author folds."""
    rng = np.random.default_rng(8)
    y = np.tile(np.array([[0] * 5, [1] * 5]), (8, 1))
    folds = np.repeat([0, 1], 8)
    x = rng.normal(size=(16, 2, 12, 15))
    c = np.zeros((16, 2, 12), int)
    for u in range(16):
        for v in (0, 1):
            c[u, v, : 1 + (u // 2 + 2 * v) % 4] = 1
    return x, c, y, folds


def test_inherited_D_and_literal_complete_fields():
    x, c, y, folds = sample()
    m = M.observations(x, c)
    np.testing.assert_allclose(
        m["weighted"], S.moments(x, c)["operators"][:, :, 2], atol=1e-14
    )
    f = M.fields(m["a"], m["geometry"], m["support"], y, folds)["fields"]
    np.testing.assert_allclose(
        f, literal_fields(literal_observations(x, c), y, folds), atol=1e-14
    )
    np.testing.assert_allclose(f[:, :, :, :4].sum(3), f[:, :, :, 4], atol=1e-14)
    terms = M.cross_terms(f)
    total = next(
        r["signed_product"]
        for r in M.statistics(f)
        if (r["kind"], r["target"]) == ("full", "all")
    )
    np.testing.assert_allclose(
        sum(r["signed_product"] for r in terms if r["target"] == "all"),
        total,
        atol=1e-14,
    )


def test_full_profile_two_view_pairing_invariance():
    x, c, y, folds = sample()
    m = M.observations(x, c)
    order, diag = M.coupling_permutation(
        y, folds, m["support"], np.random.default_rng(2)
    )
    np.testing.assert_array_equal(y[order], y)
    np.testing.assert_array_equal(folds[order], folds)
    assert diag["movable_authors"] == 16
    f = M.fields(m["a"], m["geometry"], m["support"], y, folds)["fields"]
    swapped = M.fields(m["a"][order], m["geometry"], m["support"], y, folds)["fields"]
    np.testing.assert_allclose(
        f[:, :, :, [1, 2, 5]], swapped[:, :, :, [1, 2, 5]], atol=1e-14
    )
    assert np.linalg.norm(f[:, :, :, 4] - swapped[:, :, :, 4]) > 1e-10


def test_local_supported_and_absent_accounts():
    ref = M.reference(
        np.array([0.08, -0.08]), np.array([np.eye(2), 2 * np.eye(2)]), np.ones(2, bool)
    )
    parts = M.local_account(ref, 0.04, 3 * np.eye(2), True)
    np.testing.assert_allclose(
        parts.sum(0), 0.04 * 3 * np.eye(2) - ref["total"], atol=1e-14
    )
    parts = M.local_account(ref, 0, np.full((2, 2), np.nan), False)
    np.testing.assert_allclose(parts.sum(0), -ref["total"])
    np.testing.assert_array_equal(parts[1:], 0)


def test_cold_address_guards_and_fixed_reference():
    basis = np.eye(1024, 15)
    ref = M.reference(
        np.array([0.08, -0.08]),
        np.array([np.eye(15), 2 * np.eye(15)]),
        np.ones(2, bool),
    )
    refs = {k: np.asarray([ref[k], ref[k]]) for k in ("q", "A", "G", "L", "f", "total")}
    bank = np.ones((2, 5, 15, 15))
    row = {
        "frame_key": "opaque",
        "events": [
            {
                "event_id": str(i),
                "slot": i,
                "vector": (np.zeros(1024) + i).tolist(),
                "condition": 0,
            }
            for i in range(3)
        ],
    }
    got = M.cold_record(bank, refs, basis, basis, row)
    assert got["conditional_geometry"] is None
    np.testing.assert_allclose(
        sum(np.asarray(got["readings"][name]) for name in M.COMPONENTS),
        got["relative_full"],
        atol=1e-14,
    )
    modified = copy.deepcopy(row)
    modified["events"][0]["score"] = 1
    with pytest.raises(ValueError):
        M.cold_record(bank, refs, basis, basis, modified)
    modified = copy.deepcopy(row)
    modified["events"][1]["slot"] = 0
    with pytest.raises(ValueError):
        M.cold_record(bank, refs, basis, basis, modified)


def test_zero_world_all_accounts_zero():
    x, c, y, folds = sample()
    m = M.observations(np.zeros_like(x), c)
    bank = M.fields(m["a"], m["geometry"], m["support"], y, folds)["fields"]
    assert all(row["signed_product"] == 0 for row in M.statistics(bank))
