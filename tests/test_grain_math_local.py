#!/usr/bin/env python3
"""Synthetic checks of group, qualification, source isolation and local scoring."""

from itertools import product

import numpy as np
import pytest

from grain_research.construction import math_local as L


def test_character_orthogonality_and_target_reconstruction():
    bits = np.array(list(product((-1, 1), repeat=4)))
    phi = np.column_stack([np.ones(len(bits)), L.characters(bits)])
    np.testing.assert_allclose(phi.T @ phi, 16 * np.eye(16))
    y = L.ideal_contribution(bits)
    np.testing.assert_allclose(phi @ (phi.T @ y / 16), y)
    assert np.count_nonzero(phi.T @ y / 16) == 8


def test_balanced_design_can_alias_high_order_target():
    witness = L.combinatorial_probe()
    assert witness["first_order_imbalance"] == 0
    assert witness["second_order_imbalance"] == 0
    assert witness["observed_function_difference"] == 0
    assert witness["unobserved_function_difference"] == 0.5
    assert witness["half_rank"] == 8


def test_averaging_target_operations_destroys_signal():
    bits = np.array(list(product((-1, 1), repeat=5)))
    y = L.ideal_contribution(bits)[:, 0]
    assert np.mean(y) == 0
    for b in product((-1, 1), repeat=4):
        m = np.all(bits[:, :4] == b, axis=1)
        assert np.ptp(y[m]) == 0


def test_missing_evidence_is_null_not_zero():
    result = L.local_profile([{"event_id": "a"}], np.array([0.1]), np.array([False]))
    assert result["score"] is None
    assert result["observed_events"] == 0


def test_local_event_binding_and_alias():
    events = [{"event_id": "a"}, {"event_id": "b"}, {"event_id": "a"}]
    result = L.local_profile(
        events, np.array([-1.0, 1.0, -1.0]), np.array([True, False, True])
    )
    assert result["score"] == -1
    assert result["observed_events"] == 1
    assert result["declared_events"] == 2


def test_new_own_event_changes_profile():
    result = L.local_profile(
        [{"event_id": "a"}, {"event_id": "b"}],
        np.array([-1.0, 1.0]),
        np.array([True, True]),
    )
    assert result["score"] == 0
    assert result["observed_events"] == 2


def test_conflicting_duplicate_requires_revision():
    with pytest.raises(ValueError, match="revision"):
        L.local_profile(
            [{"event_id": "a"}, {"event_id": "a"}],
            np.array([-1.0, 1.0]),
            np.array([True, True]),
        )


def test_source_fit_does_not_read_test_vectors_or_codes():
    bits = np.tile(np.array(list(product((-1, 1), repeat=5))), (5, 1))
    family = np.repeat(np.arange(5), 32)
    role = np.repeat(["source", "source", "source", "validation", "test"], 32)
    rng = np.random.default_rng(51)
    x = np.c_[bits[:, :4], bits[:, 4], rng.normal(0, 0.1, (len(bits), 2))]
    first, _ = L.fit_programs(x, bits, family, role)
    modified_x, modified_bits = x.copy(), bits.copy()
    modified_x[role == "test"] = 1e6
    modified_bits[role == "test"] *= -1
    second, _ = L.fit_programs(modified_x, modified_bits, family, role)
    for group in first:
        for key in first[group]:
            np.testing.assert_array_equal(first[group][key], second[group][key])


def test_numeric_cold_joint_unit_and_factor_classification():
    bits = np.tile(np.array(list(product((-1, 1), repeat=5))), (5, 1))
    family = np.repeat(np.arange(5), 32)
    role = np.repeat(["source", "source", "source", "validation", "test"], 32)
    x = bits.astype(float)
    models, _ = L.fit_programs(x, bits, family, role)
    scores, z = L.score_programs(models, x[role == "test"])
    report = L.metrics(scores["joint"], z, bits[role == "test"])
    assert report["unit_rmse"] < 0.01
    assert report["qualification_tp"] == 4
    assert report["qualification_fp"] == 0


def test_exact_coordinate_group_closes():
    bits = np.tile(np.array(list(product((-1, 1), repeat=5))), (3, 1))
    family = np.repeat(np.arange(3), 32)
    role = np.repeat(["source", "source", "test"], 32)
    rows = L.group_diagnostics(bits[:, :4].astype(float), bits, family, role)
    assert max(r.get("involution_defect", 0) for r in rows) < 1e-4
    assert max(r.get("commutation_defect", 0) for r in rows) < 1e-6
