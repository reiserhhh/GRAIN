#!/usr/bin/env python3
"""Synthetic checks of target-relative complements, not psychological validity."""

import numpy as np
import pytest

from grain_research.construction import math_target_complement as M


def test_exact_split_and_coordinate_reader():
    rng = np.random.default_rng(184)
    a, d, x = rng.normal(size=(7, 3)), rng.normal(size=(2, 7)), rng.normal(size=(11, 7))
    q, r = M.fixed_basis(a)
    fields = M.split_fields(d, q)
    beta = M.coordinate_fields(d, q, r)
    np.testing.assert_allclose(fields[:, 0], fields[:, 1:].sum(1), atol=2e-14)
    np.testing.assert_allclose(x @ fields[:, 1].T, (x @ a) @ beta.T, atol=2e-14)
    np.testing.assert_allclose(fields[:, 2] @ a, 0, atol=2e-14)


def test_complementary_signal_does_not_disappear():
    a = np.eye(3)[:, :2]
    q, _ = M.fixed_basis(a)
    fields = M.split_fields(np.array([0.0, 0.0, 2.0]), q)
    assert np.array_equal(fields[1], np.zeros(3))
    assert fields[2, 2] == 2


def test_fully_retained_field():
    a = np.eye(3)[:, :2]
    q, _ = M.fixed_basis(a)
    fields = M.split_fields(np.array([1.0, 2.0, 0.0]), q)
    np.testing.assert_array_equal(fields[2], np.zeros(3))


def test_original_coordinate_basis_change():
    rng = np.random.default_rng(185)
    a, d = rng.normal(size=(8, 3)), rng.normal(size=8)
    c = np.array([[2, 1, 0], [0, 3, 0], [0, 1, 4]])
    q0, _ = M.fixed_basis(a)
    q1, _ = M.fixed_basis(a @ c)
    np.testing.assert_allclose(M.split_fields(d, q0), M.split_fields(d, q1), atol=2e-14)


def test_rank_failure_not_silently_truncated():
    with pytest.raises(ValueError, match="not independent"):
        M.fixed_basis(np.ones((5, 2)))


def observation():
    """Produce a strict numeric frame with one independent evidence event."""
    return {
        "frame_key": "opaque",
        "events": [{"event_id": "e0", "slot": 0, "vector": [1.0] * 1024}],
    }


def test_cold_exact_value_and_missing():
    f, s = np.ones((2, 5, 4, 3, 1024)), np.ones((3, 1024))
    row = M.cold_record(f, s, observation())
    assert np.asarray(row["source"]).shape == (2, 5, 4, 3)
    assert row["semantic"] == [1024.0] * 3
    row = M.cold_record(f, s, {"frame_key": "empty", "events": []})
    assert row["source"] is None and row["count"] == 0


def test_labels_rejected():
    row = observation() | {"label": 1}
    with pytest.raises(ValueError, match="Only addressed"):
        M.cold_record(np.ones((2, 5, 4, 3, 1024)), np.ones((3, 1024)), row)


def test_duplicate_evidence_rejected():
    row = observation()
    row["events"] *= 2
    with pytest.raises(ValueError, match="Repeated"):
        M.cold_record(np.ones((2, 5, 4, 3, 1024)), np.ones((3, 1024)), row)


def test_perturbation_account_not_probability():
    a = np.eye(4)[:, :2]
    q, _ = M.fixed_basis(a)
    fields = M.split_fields(np.array([1, 2, -3, 4]), q)
    change = np.array([0, 1, 2, -1])
    readings = fields @ change
    assert readings[0] == readings[1] + readings[2] == -8
