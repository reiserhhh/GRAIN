#!/usr/bin/env python3
"""Conditional geometry sanity worlds; no natural psychological claim from these tests."""

import numpy as np
import pytest

from grain_research.construction import math_condition_pairing as P
from grain_research.construction import math_condition_shape as M
from scripts.run_grain_condition_shape_hm187 import literal_moments


def test_same_bag_mean_and_pairing_different_shape():
    x = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    a = M.moments(x, np.array([0, 0, 1, 1]))
    b = M.moments(x, np.array([1, 1, 0, 0]))
    for name in ("mean", "p", "covariance"):
        np.testing.assert_allclose(a[name], b[name])
    np.testing.assert_allclose(a["operators"][0], b["operators"][0])
    np.testing.assert_allclose(a["operators"][3], -b["operators"][3])
    np.testing.assert_allclose(a["operators"][3], np.diag([-0.25, 0.25]))
    assert np.linalg.eigvalsh(a["operators"][3]).min() < 0


def test_mean_shift_is_not_shape():
    x = np.array([[0.0, 0.0], [2.0, 0.0], [7.0, 4.0]])
    c = np.array([0, 0, 1])
    m = M.moments(x, c)
    np.testing.assert_allclose(m["operators"], literal_moments(x, c), atol=1e-14)
    np.testing.assert_allclose(m["operators"][1], m["operators"][2] + m["operators"][3])
    assert not m["dispersion_support"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_direct_conditional_covariance_replay(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, 2, 12, 3))
    c = rng.integers(0, 2, size=x.shape[:-1])
    np.testing.assert_allclose(
        M.moments(x, c)["operators"], literal_moments(x, c), atol=1e-14
    )


def test_orthogonal_frame_and_translation():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(8, 2, 12, 3))
    c = rng.integers(0, 2, size=x.shape[:-1])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    a = M.moments(x, c)["operators"]
    b = M.moments(x @ q + np.array([3.0, -2.0, 1.0]), c)["operators"]
    np.testing.assert_allclose(b, q.T @ a @ q, atol=1e-14)
    np.testing.assert_allclose(
        np.sum(a[:, 0] * a[:, 1]), np.sum(b[:, 0] * b[:, 1]), atol=1e-12
    )


@pytest.mark.parametrize("value", [0, 1])
def test_missing_contrast_not_zero_delta(value):
    m = M.moments(np.arange(8).reshape(4, 2), np.full(4, value))
    np.testing.assert_array_equal(m["operators"][1:], 0)
    assert np.isnan(m["conditional_shape_delta"]).all()
    assert not m["support"] and not m["dispersion_support"]


def test_singleton_opportunity_distinguished():
    m = M.moments(np.arange(8).reshape(4, 2), np.array([0, 0, 0, 1]))
    assert m["support"] and not m["dispersion_support"]
    assert np.isfinite(m["conditional_shape_delta"]).all()


def test_known_positive_target_shape_world():
    x = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    y = np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1]), (5, 1)).T
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    c = np.array([np.tile(np.array([0, 0, 1, 1]) ^ row[0], (2, 1)) for row in y])
    z = np.tile(x, (8, 2, 1, 1))
    ops = M.moments(z, c)["operators"]
    stats = M.statistics(M.fields(ops, y, folds)["fields"])
    assert (
        next(r for r in stats if r["kind"] == "shape" and r["target"] == "all")[
            "signed_product"
        ]
        > 0
    )
    np.testing.assert_allclose(P.moments(z, c)["covariance"], 0)
    zero = M.fields(M.moments(np.zeros_like(z), c)["operators"], y, folds)["fields"]
    assert all(r["signed_product"] == 0 for r in M.statistics(zero))


def test_author_matching_and_unsupported():
    x = np.zeros((5, 2, 2, 2))
    for u in range(5):
        x[u, :, 0, 0] = u - 2
    counts = np.array([1, 1, 2, 2, 3])
    a = M.author_comparison(x, counts)
    assert a["support"].tolist() == [True, True, True, True, False]
    assert np.isnan(a["stranger"][-1])
    order = M.permute_authors(counts, np.random.default_rng(6))
    np.testing.assert_array_equal(counts[order], counts)
    M.author_comparison(x, counts, order)
    with pytest.raises(ValueError):
        M.author_comparison(x, counts, [4, 0, 1, 2, 3])


def test_relink_keeps_level_and_pooled_covariance():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(8, 2, 12, 3))
    c = rng.integers(0, 2, size=x.shape[:-1])
    a = M.moments(x, c)
    b = M.moments(x, P.relink_conditions(c, rng))
    for name in ("mean", "p", "counts"):
        np.testing.assert_array_equal(a[name], b[name])
    np.testing.assert_array_equal(a["operators"][:, :, 0], b["operators"][:, :, 0])


def test_cold_no_labels_duplicate_and_missing():
    basis = np.eye(1024, 15)
    bank = np.ones((4, 2, 5, 15, 15))
    row = {
        "frame_key": "opaque",
        "events": [
            {
                "event_id": "a",
                "slot": 0,
                "vector": np.zeros(1024).tolist(),
                "condition": 0,
            },
            {
                "event_id": "b",
                "slot": 1,
                "vector": np.ones(1024).tolist(),
                "condition": 0,
            },
        ],
    }
    value = M.cold_record(bank, basis, basis, row)
    assert value["conditional_shape_delta"] is None
    assert value["original_GRAIN_mean_delta"] is None
    with pytest.raises(ValueError):
        M.cold_record(bank, basis, basis, row | {"label": 1})
    row["events"][1]["event_id"] = "a"
    with pytest.raises(ValueError):
        M.cold_record(bank, basis, basis, row)
