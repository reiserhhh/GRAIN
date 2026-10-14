#!/usr/bin/env python3
"""Finite positive/null worlds for the actual-vector source observer."""

import copy
import itertools

import numpy as np
import pytest

from grain_research.construction import math_vector_lift as V


def fixture():
    """Produce coarse target isolation with a readable fine-other-target action."""
    y = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.uint8)
    cells = np.tile([0, 1, 0, 1], (2, 3))
    f, d = np.zeros((2, 8, 8, 8)), np.zeros((5, 2, 8, 8, 8))
    for g, a, b, c in itertools.product(range(2), repeat=4):
        f[g, a, b, c] = 1 / 16
        d[0, g, a, b, c] = (-1.0) ** (g + a + b + c) / 512
    x, h = np.zeros((2, 12, 15)), np.zeros((2, 12, 1024))
    for u in range(2):
        x[u, :, 0] = (-1.0) ** (u + cells[u])
        h[u, :, 0] = x[u, :, 0]
    return x, h, cells, y, y, f, d


def bundle():
    """Create a nonconstant-input reader fixture without author/target leakage."""
    w = np.zeros((2, 5, 4096))
    w[:, :, 0] = 1
    return {
        "version": V.VERSION,
        "code_hash": V.code_hash(),
        "targets": list(V.TARGETS),
        "dimensions": 15,
        "witnesses": w.tolist(),
        "interpretation": "source_signed_tensor_reading;not_probability_or_text_intervention",
    }


def record():
    """Offer one actual event in each addressed role."""
    return {
        "frame_key": "frame",
        "view": 0,
        "events": [
            {"event_id": str(i), "slot": i, "coordinates": [0.0] * 15}
            for i in (0, 4, 8)
        ],
    }


def test_fine_crosstarget_action_survives_coarse_isolation():
    r = V.source_moments(*fixture())
    assert np.max(np.abs(r["coarse"][:, 2:])) < 1e-15
    assert np.max(np.abs(r["moments"][0, 2])) > 0.001
    assert abs(r["lifted"].sum() - 1) < 1e-15


def test_actual_support_pushforward():
    _x, _h, c, y, p, f, d = fixture()
    w, lifted, flat, _ = V.lift_weights(c, y, p, f, d)
    actual = np.bincount(
        flat.ravel(), weights=lifted.ravel(), minlength=f.size
    ).reshape(f.shape)
    np.testing.assert_allclose(actual, f - d.sum(0), atol=1e-15)
    assert (w[:, :, 1:] == 0).all()


def test_zero_action_and_coarse_only_feature_null():
    x, h, c, y, p, f, d = fixture()
    z = V.source_moments(x, h, c, y, p, f, d * 0)
    assert np.max(np.abs(z["moments"])) == 0
    x[:, :, 0] = c
    r = V.source_moments(x, h, c, y, p, f, d)
    assert np.max(np.abs(r["moments"][0, 2:])) < 1e-15


def test_all_tensor_sectors_retained():
    assert np.bincount(V.sectors()).tolist() == [1, 45, 675, 3375]


def test_subset_tensor_equals_literal_triples():
    rng = np.random.default_rng(27)
    x = rng.normal(size=(12, 15))
    phi, count, _ = V.subset_feature(x, np.arange(12))
    direct = (
        sum(
            np.kron(np.kron(np.r_[1, x[a]], np.r_[1, x[b]]), np.r_[1, x[c]])
            for a, b, c in V.OFFSETS
        )
        / 64
    )
    np.testing.assert_allclose(phi, direct, atol=1e-14)
    assert count == 64


def test_cold_observed_and_unobserved():
    r = record()
    observed = V.cold_record(bundle(), r)
    assert observed["triples"] == 1
    np.testing.assert_allclose(observed["retained_readings"], 1 / 64)
    np.testing.assert_allclose(observed["conditional_readings"], 1)
    r["events"] = r["events"][:2]
    empty = V.cold_record(bundle(), r)
    assert empty["conditional_readings"] is None
    assert np.max(np.abs(empty["retained_readings"])) == 0


@pytest.mark.parametrize(
    "kind",
    ["label", "duplicate_id", "duplicate_slot", "float_slot", "nan", "extra_bundle"],
)
def test_cold_rejects_incorrect_evidence(kind):
    r, b = copy.deepcopy(record()), bundle()
    if kind == "label":
        r["events"][0]["target"] = 1
    elif kind == "duplicate_id":
        r["events"][1]["event_id"] = r["events"][0]["event_id"]
    elif kind == "duplicate_slot":
        r["events"][1]["slot"] = r["events"][0]["slot"]
    elif kind == "float_slot":
        r["events"][0]["slot"] = 0.5
    elif kind == "nan":
        r["events"][0]["coordinates"][0] = float("nan")
    else:
        b["application_labels"] = []
    with pytest.raises(ValueError):
        V.cold_record(b, r)


def test_missing_source_support_is_not_filled():
    _x, _h, c, y, p, f, d = fixture()
    f[0, 0, 0, 0] = 0
    with pytest.raises(ValueError, match="support"):
        V.lift_weights(c, y, p, f, d)
