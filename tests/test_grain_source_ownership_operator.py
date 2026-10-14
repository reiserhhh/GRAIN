#!/usr/bin/env python3
"""Exact group laws must coexist with independent cold-action tests."""

import numpy as np
import pytest

from grain_research.construction import math_source_ownership_operator as M


def toy():
    even = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    odd = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [-1.0, -1.0, 0.0]])
    return even + odd, even - odd


def test_oblique_source_and_single_input():
    plus, minus = toy()
    models, info = M.fit(plus, minus)
    assert info["oblique_constructed"]
    model = models["oblique"]
    for h, partner in zip(plus, minus, strict=True):
        np.testing.assert_allclose(M.operate(model, h), partner, atol=1e-12)
        np.testing.assert_allclose(M.operate(model, M.operate(model, h)), h, atol=1e-12)
    with pytest.raises(ValueError, match="Exactly one"):
        M.operate(model, plus)


def test_orthogonal_reflection_is_not_semantic_exchange():
    plus, minus = toy()
    models, _ = M.fit(plus, minus)
    orth = M.summarize(M.pair_accounts(models["orthogonal"], plus, minus))
    oblique = M.summarize(M.pair_accounts(models["oblique"], plus, minus))
    assert orth["even_leakage_error_rms"] > 0.1
    assert oblique["relative_error"] < 1e-12


@pytest.mark.parametrize("method", M.METHODS)
def test_residual_identity(method):
    plus, minus = toy()
    model = M.fit(plus, minus)[0][method]
    rng = np.random.default_rng(17)
    a, b = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    accounts = M.pair_accounts(model, a, b)
    direct = (
        np.sum((np.array([M.operate(model, h) for h in a]) - b) ** 2, axis=1)
        + np.sum((np.array([M.operate(model, h) for h in b]) - a) ** 2, axis=1)
    ) / 2
    np.testing.assert_allclose(direct, accounts["operator_error_sq"], atol=1e-12)


def test_source_perfect_does_not_imply_cold_semantics():
    plus, minus = toy()
    model = M.fit(plus, minus)[0]["oblique"]
    even = np.array([[0.0, 1.0, 0.0]])
    odd = np.array([[0.0, 0.0, 1.0]])
    cold = M.summarize(M.pair_accounts(model, even + odd, even - odd))
    assert cold["relative_error"] > 1
    np.testing.assert_allclose(
        M.operate(model, M.operate(model, even[0])), even[0], atol=1e-12
    )


def test_intersection_is_recorded_not_regularized():
    even = np.array([[1.0, 0.0], [-1.0, 0.0]])
    odd = np.array([[1.0, 0.0], [2.0, 0.0]])
    models, info = M.fit(even + odd, even - odd)
    assert not info["oblique_constructed"]
    assert set(models) == {"orthogonal"}


def test_invalid_sources():
    with pytest.raises(ValueError, match="finite"):
        M.fit(np.array([[np.nan]]), np.ones((1, 1)))
    with pytest.raises(ValueError, match="Comparable"):
        M.fit(np.ones((1, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="No source"):
        M.fit(np.ones((2, 2)), np.ones((2, 2)))


def test_fresh_documents_complete_and_exact_reversal():
    spec = {
        "anchor": "opened a book",
        "behaviours": {
            q: ["performed low " + q, "performed high " + q] for q in M.TARGETS
        },
    }
    docs = M.fresh_documents(spec)
    assert len(docs) == 96
    assert len({d["body"] for d in docs}) == 96
    for i in range(6):
        for j in range(8):
            assert (
                docs[16 * i + j]["body"].splitlines()
                == docs[16 * i + 8 + j]["body"].splitlines()[::-1]
            )


def test_collector_path_guard_and_launcher(tmp_path):
    from scripts.run_grain_source_ownership_operator import OUT
    from scripts.run_grain_upstream_routes import collector_guard

    assert hasattr(OUT, "parent")
    out = tmp_path / "packet"
    lock = tmp_path / "packet.lock"
    with collector_guard(out):
        assert lock.exists()
        with pytest.raises(FileExistsError), collector_guard(out):
            pass
    assert not lock.exists()


def test_asset_enumeration_excludes_appledouble(tmp_path):
    from scripts.run_grain_source_ownership_operator import files

    (tmp_path / "model.npz").touch()
    (tmp_path / "._model.npz").touch()
    (tmp_path / "nested.npz").mkdir()
    assert [p.name for p in files(tmp_path, "*.npz")] == ["model.npz"]
