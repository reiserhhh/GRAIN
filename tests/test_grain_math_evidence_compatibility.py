#!/usr/bin/env python3
"""Distinguish contextual direction, scalar modulation and output-link artifacts."""

import numpy as np
import pytest

from grain_research.construction import math_evidence_compatibility as M


def test_complete_square_has_no_boundary_circulation():
    values = np.random.default_rng(1).normal(size=(8, 4))
    edges, _ = M.conditional_edges(values, 3)
    rows = M.square_rows(values, edges.mean(1), 3)
    assert len(rows) == 6
    assert max(r["compatibility_error"] for r in rows) < 1e-14


def test_additive_distinct_lines_have_no_interaction_or_rotation():
    values = M.synthetic_world("additive", 2, 3)["native"]
    edges, _ = M.conditional_edges(values, 3)
    assert all(M.line_split(v, v.mean(0))["normal_fraction"] < 1e-25 for v in edges)
    rows = M.square_rows(values, edges.mean(1), 3)
    assert max(r["interaction_energy"] for r in rows) < 1e-25


def test_common_scalar_nonlinearity_can_interact_without_direction_change():
    values = M.synthetic_world("common_scalar_nonlinear", 3, 3)["native"]
    edges, _ = M.conditional_edges(values, 3)
    rows = M.square_rows(values, edges.mean(1), 3)
    assert max(r["interaction_energy"] for r in rows) > 0.01
    assert max(M.line_split(v, v.mean(0))["normal_fraction"] for v in edges) < 1e-25
    assert max(r["separation"] for r in rows) < 1e-14


def test_selective_interaction_forces_nonzero_projection_bound():
    values = M.synthetic_world("selective_interaction", 4, 3)["native"]
    edges, _ = M.conditional_edges(values, 3)
    row = next(
        r for r in M.square_rows(values, edges.mean(1), 3) if (r["j"], r["k"]) == (0, 1)
    )
    assert row["angle_bound"] > 1
    assert row["projection_bound"] >= row["angle_bound"] - 1e-14
    assert row["four_edge_normal_energy"] >= row["projection_bound"] - 1e-14


def test_four_edge_bound_is_attainable_for_centered_interaction():
    values = np.column_stack([M.corners(2), 0.5 * np.prod(M.corners(2), axis=1)])
    edges, _ = M.conditional_edges(values, 2)
    row = M.square_rows(values, edges.mean(1), 2)[0]
    assert row["four_edge_normal_energy"] == pytest.approx(row["projection_bound"])


def test_output_link_rotation_is_not_preanswer_rotation():
    world = M.synthetic_world("link_only", 5, 3)
    values = M.table(world["preanswer"], 3)
    edges, _ = M.conditional_edges(values, 3)
    assert max(M.line_split(v, v.mean(0))["normal_fraction"] for v in edges) < 1e-25
    edges, _ = M.conditional_edges(world["native"], 3)
    assert max(M.line_split(v, v.mean(0))["normal_fraction"] for v in edges) > 1e-4


def test_orthogonal_coordinate_change_preserves_energy_and_angle_bounds():
    rng = np.random.default_rng(6)
    values = rng.normal(size=(8, 4))
    rotation = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    edges, _ = M.conditional_edges(values, 3)
    a = M.square_rows(values, edges.mean(1), 3)
    b = M.square_rows(values @ rotation, edges.mean(1) @ rotation, 3)
    for r, s in zip(a, b, strict=True):
        for key in (
            "interaction_energy",
            "projection_bound",
            "angle_bound",
            "four_edge_normal_energy",
        ):
            assert r[key] == pytest.approx(s[key])


def test_negative_gains_are_retained_as_same_line():
    d = np.array([[2.0, 0], [-3.0, 0]])
    split = M.line_split(d, np.array([1.0, 0]))
    assert split["normal_energy"] == 0
    assert split["negative_gain_count"] == 1


def test_zero_mean_cancellation_is_not_assigned_a_direction():
    d = np.array([[2.0, 0], [-2.0, 0]])
    assert not M.line_split(d, d.mean(0))["identified"]


def test_independent_scalar_modulations_can_violate_state_compatibility():
    dj_minus, dj_plus = np.array([1.0, 0]), np.array([2.0, 0])
    dk_minus, dk_plus = np.array([0.0, 1]), np.array([0.0, 1])
    assert np.linalg.norm((dj_plus - dj_minus) - (dk_plus - dk_minus)) == 1


@pytest.mark.parametrize("values", [np.zeros((3, 2)), np.full((4, 2), np.nan)])
def test_incomplete_or_nonfinite_observations_fail(values):
    with pytest.raises(ValueError):
        M.conditional_edges(values, 2)


def test_unknown_world_fails():
    with pytest.raises(ValueError):
        M.synthetic_world("unmeasured", 1)
