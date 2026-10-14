#!/usr/bin/env python3
"""Nested polynomial/cell geometry, contraction and cohort checks."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms import grain_cell_residual as f
from grain_research.vector_mechanisms import joint_moment_nullspace as previous


def toy(seed=731):
    """Repeated exact rows leave a nontrivial within-cell vector space."""
    rng = np.random.default_rng(seed)
    g = np.repeat(rng.integers(-2, 3, size=(18, 4)), 3, axis=0).astype(float)
    return g, rng.normal(size=(4, len(g), 7)), rng.normal(size=(2, len(g), 7))


def test_nesting_and_all_component_identities():
    g, ds, jj = toy()
    maps = f.CellMaps(g)
    pieces = maps.decompose(jj[0])
    np.testing.assert_allclose(sum(pieces), jj[0], atol=1e-12)
    for i in range(4):
        for j in range(i):
            assert np.sum(pieces[i] * pieces[j]) == pytest.approx(0, abs=1e-10)
    rows, checks = f.analyze(g, ds, jj)
    assert max(checks.values()) < 1e-10
    assert [len(rows[k]) for k in f.TABLES] == [1, 16, 64, 96, 4]
    _, old_q, _ = previous.basis(g)
    np.testing.assert_allclose(pieces[0], old_q @ (old_q.T @ jj[0]), atol=1e-11)


def test_conditional_covariance_is_explicit_within_cell_action():
    g, ds, jj = toy()
    maps = f.CellMaps(g)
    expected = 0.0
    for cell in np.unique(maps.index):
        j, d = jj[0, maps.index == cell], ds[0, maps.index == cell]
        expected += np.sum((j - j.mean(0)) * (d - d.mean(0)))
    assert maps.covariance_action(jj[0], ds[0]) == pytest.approx(expected, abs=1e-11)


def test_degenerate_and_singleton_cells_are_not_rebinned():
    g, ds, jj = toy()
    maps = f.CellMaps(np.zeros_like(g))
    assert maps.geometry["linear_rank"] == maps.geometry["quadratic_rank"] == 1
    assert maps.geometry["cells"] == 1
    unique = np.column_stack((np.arange(len(g)), np.zeros((len(g), 3))))
    maps = f.CellMaps(unique)
    assert maps.geometry["singleton_token_fraction"] == 1
    np.testing.assert_array_equal(maps.decompose(jj[0])[3], np.zeros_like(jj[0]))
    zero, checks = f.analyze(unique, ds, np.zeros_like(jj))
    assert max(checks.values()) < 1e-10
    assert all(np.isnan(row["cosine"]) for row in zero["query_geometry"])


def test_signed_scalar_cancellation_survives_vector_orthogonality():
    g, ds, jj = toy()
    parts = f.CellMaps(g).decompose(jj[0])
    ds[0] = parts[0] / np.sum(parts[0] ** 2) - parts[3] / np.sum(parts[3] ** 2)
    rows, _ = f.analyze(g, ds, jj)
    values = [
        r
        for r in rows["responses"]
        if r["query"] == "neuroticism" and r["factor"] == "F01"
    ]
    assert values[0]["action"] == pytest.approx(1, abs=1e-11)
    assert values[3]["action"] == pytest.approx(-1, abs=1e-11)
    assert abs(values[0]["native"]) < 1e-11


@pytest.mark.parametrize("sign,expected", [(1, 1), (-1, 0)])
def test_common_and_opposing_query_geometry(sign, expected):
    g, ds, jj = toy()
    rows, _ = f.analyze(g, ds, np.stack([jj[0], sign * jj[0]]))
    for row in rows["query_geometry"]:
        assert row["common_energy"] / row["pair_energy"] == pytest.approx(
            expected, abs=1e-11
        )


def test_hidden_rotation_and_token_permutation_invariance():
    g, ds, jj = toy()
    rng = np.random.default_rng(88)
    rotate = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    order = rng.permutation(len(g))
    original, _ = f.analyze(g, ds, jj)
    changed, _ = f.analyze(g[order], ds[:, order] @ rotate, jj[:, order] @ rotate)
    for name in f.TABLES:
        pd.testing.assert_frame_equal(
            pd.DataFrame(original[name]),
            pd.DataFrame(changed[name]),
            atol=1e-10,
            rtol=1e-10,
        )


def test_all_cohorts_queries_factors_and_cross_terms_are_published():
    raw = {k: [] for k in f.TABLES}
    for i in range(12):
        rows, _ = f.analyze(*toy(731 + i))
        for name in f.TABLES:
            raw[name].extend(
                {"cohort": "source4" if i < 4 else "new8", "author_key": str(i), **r}
                for r in rows[name]
            )
    table = f.aggregate({k: pd.DataFrame(v) for k, v in raw.items()})
    assert [len(table[k]) for k in f.TABLES] == f.COUNTS
    assert set(table["responses"].factor) == {"all4", "F01", "F02", "F03", "F04"}
