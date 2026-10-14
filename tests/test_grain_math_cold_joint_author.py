#!/usr/bin/env python3
"""Finite cold observation tests; no natural labels or neural calls."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_cold_joint_author as M


@pytest.fixture
def bundle():
    rng = np.random.default_rng(8101)
    return M.source_bundle(
        rng.normal(size=(32, 4)), rng.uniform(size=(32, 6)), {}, "fixed", {}
    )


def record():
    return {
        "author_key": "u",
        "events": [
            {
                "event_id": str(j),
                "condition": "ordinary" if j < 2 else "deadline",
                "qualification": "focal_report",
                "observer_id": "fixed",
                "grain": [j, j / 2, -j, 1],
                "query": [j / 4] * 6,
            }
            for j in range(4)
        ],
    }


def test_joint_before_mean():
    g, q = np.tile([-1.0, 1.0], (4, 1)).T, np.tile([-1.0, 1.0], (6, 1)).T
    assert np.all(M.lift(g, q).mean(0)[10:34] == 1)
    assert np.all(np.outer(g.mean(0), q.mean(0)) == 0)


def test_permuted_pairing_preserves_marginals():
    g, q = np.arange(16.0).reshape(4, 4), np.arange(24.0).reshape(4, 6)
    a, b = M.lift(g, q), M.lift(g, q[::-1])
    np.testing.assert_array_equal(a.mean(0)[:10], b.mean(0)[:10])
    assert not np.allclose(a.mean(0)[10:34], b.mean(0)[10:34])


def test_kernel_identity():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(5, 49)), rng.normal(size=(7, 49))
    direct = np.sum((a.mean(0) - b.mean(0)) ** 2)
    gram = (a @ a.T).mean() + (b @ b.T).mean() - 2 * (a @ b.T).mean()
    assert direct == pytest.approx(gram)


def test_contributions_conditions_covariance(bundle):
    rec = record()
    out = M.cold_score(bundle, rec)
    np.testing.assert_allclose(
        np.sum([r["contribution"] for r in out["contributions"]], axis=0),
        out["profile"],
    )
    weighted = sum(
        c["share"] * np.asarray(c["profile"])
        for c in out["conditional_profiles"].values()
    )
    np.testing.assert_allclose(weighted, out["profile"])
    assert all(
        c["outside_source_condition_support"]
        for c in out["conditional_profiles"].values()
    )
    _, g, q = M.observed_features(
        bundle, [e["grain"] for e in rec["events"]], [e["query"] for e in rec["events"]]
    )
    np.testing.assert_allclose(
        out["grain_query_covariance"],
        (g - g.mean(0)).T @ (q - q.mean(0)) / len(g),
        atol=1e-12,
    )


def test_batch_and_order_independent(bundle):
    a = record()
    b = copy.deepcopy(a)
    b["events"].reverse()
    np.testing.assert_allclose(
        M.cold_score(bundle, a)["profile"], M.cold_score(bundle, b)["profile"]
    )
    assert M.cold_score(bundle, a) == M.cold_score(bundle, a)


def test_unobserved_is_not_zero(bundle):
    a = record()
    for event in a["events"]:
        event["qualification"] = "unresolved"
    assert M.cold_score(bundle, a)["profile"] is None
    assert M.cold_score(bundle, a)["qualification_counts"]["unresolved"] == 4


def test_duplicate_and_truth_rejected(bundle):
    a = record()
    a["events"].append(a["events"][0])
    with pytest.raises(ValueError):
        M.cold_score(bundle, a)
    a = record()
    a["truth"] = 1
    with pytest.raises(ValueError):
        M.cold_score(bundle, a)


def test_bundle_drift_rejected(bundle):
    bundle["query_center"][0] += 1
    with pytest.raises(ValueError):
        M.cold_score(bundle, record())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 2.0])
def test_invalid_query_rejected(bundle, bad):
    a = record()
    a["events"][0]["query"][0] = bad
    with pytest.raises(ValueError):
        M.cold_score(bundle, a)
