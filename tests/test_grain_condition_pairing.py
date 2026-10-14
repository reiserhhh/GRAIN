#!/usr/bin/env python3
"""Mechanism-discriminating toy worlds and strict numerical observation tests."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_condition_pairing as M
from grain_research.construction import math_target_complement as C


def source():
    """Give each author fold both sides of every target, without a fitted model."""
    labels = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(int)
    return np.tile(labels, (2, 1)), np.repeat([0, 1], 32)


def test_reynolds_pairing_average():
    h = np.arange(12).reshape(4, 3)
    c = np.array([0, 0, 1, 1])
    m = M.moments(h, c)
    average = np.mean(
        [M.moments(h, c[list(p)])["mixed"] for p in itertools.permutations(range(4))], 0
    )
    np.testing.assert_allclose(average, m["p"] * m["mean"], atol=1e-14)


def test_same_bag_and_choice_different_pairing():
    h = np.array([[-1.0], [-1.0], [1.0], [1.0]])
    a, b = M.moments(h, [0, 0, 1, 1]), M.moments(h, [1, 1, 0, 0])
    np.testing.assert_array_equal(a["mean"], b["mean"])
    assert a["p"] == b["p"] == 0.5
    np.testing.assert_allclose(a["covariance"], -b["covariance"])
    assert a["covariance"][0] > 0


def test_conditional_opportunity_identity():
    h = np.arange(24).reshape(6, 4)
    m = M.moments(h, [0, 1, 0, 0, 1, 0])
    np.testing.assert_allclose(m["covariance"], m["opportunity"] * m["delta"])
    np.testing.assert_allclose(m["mixed"], m["p"] * m["mean"] + m["covariance"])
    np.testing.assert_allclose(
        m["delta"], h[[1, 4]].mean(0) - h[[0, 2, 3, 5]].mean(0), atol=1e-14
    )


@pytest.mark.parametrize("value", [0, 1])
def test_missing_opportunity_not_neutral(value):
    h = np.random.default_rng(0).normal(size=(4, 7))
    m = M.moments(h, np.full(4, value))
    assert not m["support"]
    np.testing.assert_array_equal(m["covariance"], np.zeros(7))
    assert np.isnan(m["delta"]).all()


def test_simultaneous_reorder_is_invariant():
    h, c = np.arange(15).reshape(5, 3), np.array([1, 0, 1, 0, 0])
    order = [3, 2, 0, 4, 1]
    a, b = M.moments(h, c), M.moments(h[order], c[order])
    for key in a:
        np.testing.assert_allclose(a[key], b[key])


def test_fields_use_disjoint_author_groups_and_sign():
    y, folds = source()
    h = np.random.default_rng(1).normal(size=(64, 2, 9))
    a, b = M.source_fields(h, h / 2, y, folds), M.source_fields(h, h / 2, 1 - y, folds)
    np.testing.assert_allclose(a["fields"], -b["fields"])
    np.testing.assert_allclose(a["weights"].sum(-1), 0, atol=1e-14)
    assert not np.any((a["weights"][0] != 0) & (a["weights"][1] != 0))
    for q in range(5):
        ids = folds == 0
        literal = h[ids & (y[:, q] == 1)].mean(0) - h[ids & (y[:, q] == 0)].mean(0)
        np.testing.assert_allclose(a["fields"][0, 0, :, q], literal)


def test_nulls_preserve_profiles_and_opportunities():
    y, folds = source()
    rng = np.random.default_rng(2)
    yp = M.permute_profiles(y, folds, rng)
    for b in (0, 1):
        assert sorted(map(tuple, yp[folds == b])) == sorted(map(tuple, y[folds == b]))
    c = rng.integers(0, 2, size=(64, 2, 12))
    cp = M.relink_conditions(c, rng)
    np.testing.assert_array_equal(c.sum(-1), cp.sum(-1))


def test_cross_stat_literal_and_failure_world():
    fields = np.zeros((2, 2, 2, 5, 3, 7))
    rows = M.statistics(fields)
    assert len(rows) == 36
    assert all(r["signed_inner_product"] == 0 for r in rows)
    assert all(r["cross_author_coherence"] is None for r in rows)
    f = np.random.default_rng(3).normal(size=fields.shape)
    rows = M.statistics(f)
    primary = next(
        r
        for r in rows
        if (r["kind"], r["target"], r["space"]) == ("pairing", "all", "full")
    )
    literal = (
        sum(
            f[0, 1, v, q, 0] @ f[1, 1, w, q, 0]
            for v in (0, 1)
            for w in (0, 1)
            for q in range(5)
        )
        / 4
    )
    np.testing.assert_allclose(primary["signed_inner_product"], literal)


def test_known_pairing_world_succeeds_without_level_or_choice_change():
    y, folds = source()
    h = np.broadcast_to(np.array([[[-1.0], [1.0]]]), (64, 2, 2, 1)).copy()
    c = np.broadcast_to(np.array([0, 1]), (64, 2, 2)).copy()
    c[y[:, 0] == 0] = 1 - c[y[:, 0] == 0]
    m = M.moments(h, c)
    np.testing.assert_array_equal(m["mean"], np.zeros((64, 2, 1)))
    np.testing.assert_array_equal(m["p"], np.full((64, 2), 0.5))
    f = M.source_fields(m["mean"], m["covariance"], y, folds)["fields"]
    f = np.stack((f, f, np.zeros_like(f)), axis=-2)
    primary = next(
        r
        for r in M.statistics(f)
        if (r["kind"], r["target"], r["space"]) == ("pairing", "all", "full")
    )
    assert primary["signed_inner_product"] > 0
    assert primary["within_fold_cross_view"] > 0


def cold_input():
    """Build numerical addressed events without author labels or model answers."""
    return {
        "frame_key": "anonymous",
        "events": [
            {
                "event_id": str(i),
                "slot": i,
                "condition": i % 2,
                "vector": np.full(1024, i).tolist(),
            }
            for i in range(4)
        ],
    }


def test_cold_full_accounts_and_original_coordinate_reading():
    rng = np.random.default_rng(4)
    mapping = rng.normal(size=(1024, 15))
    basis, triangular = C.fixed_basis(mapping)
    full = rng.normal(size=(2, 2, 5, 1024))
    fields = C.split_fields(full, basis)
    record = cold_input()
    out = M.cold_record(fields, record)
    assert out["delta_support"] and out["condition_counts"] == [2, 2]
    beta = C.coordinate_fields(full, basis, triangular)
    x = np.array([e["vector"] for e in record["events"]])
    m = M.moments(x, [e["condition"] for e in record["events"]])
    for name in ("mean", "mixed", "covariance"):
        value = np.asarray(out["readings"][name])
        np.testing.assert_allclose(
            value[..., 0], value[..., 1] + value[..., 2], atol=1e-12
        )
        np.testing.assert_allclose(
            value[..., 1], np.einsum("kvqd,d->kvq", beta, m[name] @ mapping), atol=1e-12
        )


def test_cold_missing_and_empty():
    f, row = np.ones((2, 2, 5, 3, 1024)), cold_input()
    for e in row["events"]:
        e["condition"] = 1
    out = M.cold_record(f, row)
    assert out["readings"]["conditional_delta"] is None
    row["events"] = []
    assert M.cold_record(f, row)["readings"] is None


@pytest.mark.parametrize("mutation", ["label", "answer", "duplicate", "condition"])
def test_cold_rejects_leakage_and_bad_addresses(mutation):
    row = cold_input()
    if mutation in ("label", "answer"):
        row["events"][0][mutation] = 1
    elif mutation == "duplicate":
        row["events"].append(row["events"][0])
    else:
        row["events"][0]["condition"] = 2
    with pytest.raises(ValueError):
        M.cold_record(np.ones((2, 2, 5, 3, 1024)), row)


def test_validation_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        M.moments(np.empty((0, 4)), np.array([]))
    with pytest.raises(ValueError):
        M.source_fields(np.ones((4, 2, 3)), np.ones((4, 2, 3)), np.ones((4, 5)))
