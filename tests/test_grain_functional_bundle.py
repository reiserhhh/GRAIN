#!/usr/bin/env python3
"""Portable bundle replay and measurement-contract safeguards without real data."""

import hashlib
import json

import numpy as np
import pytest

from grain_research.vector_mechanisms.functional_bundle import FrozenFunctionalBundle


@pytest.fixture
def bundle(tmp_path):
    """Create a tiny mixed full/low-rank bundle with an explicit frozen observer."""
    queries = ["a", "b", "c"]
    arrays = {}
    for q in queries:
        data = {
            "p_ref": np.zeros((2, 3)),
            "q_ref": np.zeros(3),
            "anchor": 0.4,
            "mean_coef": np.ones(3),
            "q_coef": np.ones(3),
            "rank": 1,
        }
        if q == "a":
            data["configuration"] = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
        else:
            data.update(left=np.array([[1.0], [-1.0]]), right=np.array([[1.0, 0, 0]]))
        arrays.update({f"{q}__{k}": v for k, v in data.items()})
    np.savez(tmp_path / "compact_functions.npz", **arrays)
    (tmp_path / "author_reference.json").write_text(
        json.dumps(
            {
                "coordinates": [str(j) for j in range(9)],
                "center": [0] * 9,
                "scale": [2] * 9,
                "active": [True] * 9,
            }
        )
    )
    (tmp_path / "matched_query_projection.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "queries": queries,
                    "centers": {"0/e1": [0, 0, 0]},
                    "scale": [1, 1, 1],
                    "active": [True] * 3,
                    "coef": [[0, 0.2, 0], [0.1, 0, 0.3], [0, 0.1, 0]],
                }
            }
        )
    )
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.iterdir()
        if not p.name.startswith("._")
    }
    (tmp_path / "bundle.json").write_text(
        json.dumps(
            {
                "protocol": "GRAIN_FUNCTIONAL_BUNDLE_M26_V1",
                "files": files,
                "queries": queries,
                "question_tokens": {q: [1, 2] for q in queries},
                "observer_configuration": {"model": "synthetic"},
            }
        )
    )
    return FrozenFunctionalBundle.load(tmp_path), tmp_path


def test_full_and_low_rank_replay(bundle):
    b, _ = bundle
    p, q = np.array([[1.0, 2, 3], [4, 5, 6]]), np.ones(3)
    a = b.score("a", p, q, [1, 2], {"model": "synthetic"})
    low = b.score("b", p, q, [1, 2], {"model": "synthetic"})
    assert a == low
    assert a["score"] == pytest.approx(0.4 + 10.5 - 3 + 3)


def test_contract_and_hash_rejected(bundle):
    b, path = bundle
    with pytest.raises(ValueError, match="Observer"):
        b.score("a", np.zeros((2, 3)), np.zeros(3), [1, 2], {"model": "other"})
    with pytest.raises(ValueError, match="token"):
        b.score("a", np.zeros((2, 3)), np.zeros(3), [2, 1], {"model": "synthetic"})
    (path / "author_reference.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        FrozenFunctionalBundle.load(path)


def test_profile_and_shared_accounting(bundle):
    b, _ = bundle
    states = {
        q: {c: (np.zeros((2, 3)), np.zeros(3)) for c in ("full", "e1", "e2")}
        for q in ("a", "b", "c")
    }
    result = b.profile(states, {q: [1, 2] for q in states}, {"model": "synthetic"})
    np.testing.assert_allclose(result["raw"], [0.4, 0.4, 0.4, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(result["positions"], [0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0])
    s = b.shared_private([1, 2, 3], 0, "e1")
    np.testing.assert_allclose(np.array(s["shared"]) + s["private"], s["standardized"])
    with pytest.raises(ValueError, match="condition"):
        b.shared_private([1, 2, 3], 7, "e1")


def test_bundle_batch_independent(bundle):
    b, _ = bundle
    x = np.arange(27.0).reshape(3, 9)
    np.testing.assert_array_equal(b.coordinates(x[:1]), b.coordinates(x)[:1])
