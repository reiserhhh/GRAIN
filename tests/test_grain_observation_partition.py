#!/usr/bin/env python3
"""Observation-unit construction tests, not personality accuracy tests."""

import numpy as np
import pytest

from grain_research.construction import math_observation_partition as M
from tests.test_grain_vector_likelihood import world


@pytest.mark.parametrize("tau", [0.0, 0.7])
def test_lossless_groupings_preserve_reading(tau):
    z, y, _ = world()
    x = np.random.default_rng(205).normal(size=(8, 2))
    readings = []
    for name, a in M.programs(8).items():
        if name == "lossy_prefix":
            continue
        readings.append(M.read_grouped(z, y, a @ x, a, 1.3, tau)["posterior"])
    assert np.allclose(readings, readings[0], atol=1e-12)


def test_disjoint_independent_diagonal_is_valid():
    for name in ("atomic", "pooled", "balanced", "unequal"):
        a = M.programs(8)[name]
        joint, diag = M.geometry(a, 1.3), M.geometry(a, 1.3, diagonal=True)
        assert np.allclose(joint["root_weights"], diag["root_weights"])
        assert joint["alpha"] == pytest.approx(diag["alpha"])


def test_alias_information_not_new_evidence():
    a = M.programs(8)["aliases"]
    joint, diag = M.geometry(a, 1.3), M.geometry(a, 1.3, diagonal=True)
    assert joint["alpha"] == pytest.approx(8 / 1.3)
    assert diag["alpha"] == pytest.approx(12 / 1.3)
    assert joint["rank"] == 8


def test_shared_occasion_disjoint_groups_are_dependent():
    a = M.programs(8)["balanced"]
    joint, diag = M.geometry(a, 1.0, 0.7), M.geometry(a, 1.0, 0.7, True)
    assert diag["alpha"] > joint["alpha"]
    assert joint["alpha"] == pytest.approx(8 / (1 + 8 * 0.7))


def test_lossy_group_changes_observation_not_reexpression():
    a = M.programs(8)["lossy_prefix"]
    info = M.geometry(a, 1.0)
    assert info["alpha"] == pytest.approx(4)
    assert info["mean_recovery_defect"] > 0


def test_unitization_can_reverse_a_relative_reading():
    z = np.array([[-1.0], [1.0]])
    y = np.array([[0] * 5, [1] * 5])
    x = np.array([[-2.0], [1], [1], [1]])
    a = M.programs(4)["unequal"]
    correct = M.read_grouped(z, y, a @ x, a, 1.0, 0.7)
    naive = M.read_grouped(z, y, a @ x, a, 1.0, 0.7, True)
    assert np.all(correct["relative"] > 0)
    assert np.all(naive["relative"] < 0)


def test_literal_and_joint_profile_agree():
    z, y, _ = world()
    x = np.random.default_rng(17).normal(size=(8, 2))
    a = M.programs(8)["overlap"]
    result = M.read_grouped(z, y, a @ x, a, 1.3, 0.7)
    assert np.allclose(result["posterior"][0], M.literal(z, y, a @ x, a, 1.3, 0.7))
    bits = (np.arange(32)[:, None] // (2 ** np.arange(5))) % 2
    assert np.allclose(result["profile"] @ bits, result["posterior"])


def test_cold_provenance_accepts_aliases_not_fake_events():
    r = {
        "event_ids": ["a", "b", "c", "d"],
        "vectors": [[1, 2]] * 4,
        "grouping": M.programs(4)["aliases"].tolist(),
    }
    x, g = M.cold_observation(r, 2)
    assert len(x) == 4 and len(g) == 8
    with pytest.raises(ValueError, match="identity"):
        M.cold_observation(r | {"event_ids": ["a"] * 4}, 2)
    with pytest.raises(ValueError, match="Only"):
        M.cold_observation(r | {"personality": 1}, 2)


def test_impossible_alias_observations_rejected():
    z, y, _ = world()
    a = M.programs(4)["aliases"]
    g = a @ np.zeros((4, 2))
    g[-1, 0] = 1
    with pytest.raises(ValueError, match="support"):
        M.read_grouped(z, y, g, a, 1.0)


@pytest.mark.parametrize("sigma,tau", [(0, 0), (-1, 0), (1, -1), (np.nan, 0)])
def test_invalid_laws_rejected(sigma, tau):
    with pytest.raises(ValueError):
        M.geometry(np.eye(4), sigma, tau)
