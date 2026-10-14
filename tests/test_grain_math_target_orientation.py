#!/usr/bin/env python3
"""Synthetic controls for finite cone alternatives and author composition."""

import numpy as np
import pytest

from grain_research.construction import math_target_orientation as M


def test_positive_cone_has_primal_dual_optimum():
    d = np.array([[1, 0], [0, 1], [1, 1.0]])
    result = M.orientation(d)
    assert result["status"] == "SOURCE_ORIENTATION_EXISTS"
    assert d @ result["w"] == pytest.approx([1, 1, 2])
    assert result["L1"] == pytest.approx(2)
    assert result["certificate_error"] < 1e-12


@pytest.mark.parametrize("d", [[[1], [-1]], [[0]], [[1, 0], [-1, 0], [0, 1]]])
def test_cancellation_not_fake_zero_score(d):
    d = np.asarray(d, float)
    result = M.orientation(d)
    assert result["status"] == "NO_COMMON_ORIENTATION_ON_DECLARED_SUPPORT"
    assert result["w"] is None
    assert d.T @ result["dual"] == pytest.approx(np.zeros(d.shape[1]), abs=1e-12)
    assert sum(result["dual"]) == pytest.approx(1)


def test_separate_directions_can_exist_without_common_orientation():
    assert M.orientation([[1]])["w"] == pytest.approx([1])
    assert M.orientation([[-1]])["w"] == pytest.approx([-1])
    assert M.orientation([[1], [-1]])["w"] is None


def test_invertible_frame_preserves_existence_not_minimum_L1():
    d = np.array([[1, 0], [0, 1], [1, 1.0]])
    changed = d @ np.array([[2, 1], [0, 3.0]])
    assert M.orientation(d)["status"] == M.orientation(changed)["status"]
    assert M.orientation(d)["L1"] != pytest.approx(M.orientation(changed)["L1"])


@pytest.mark.parametrize("d", [[], [[np.nan]], [[np.inf]], [1, 2]])
def test_invalid_support_rejected(d):
    with pytest.raises(ValueError):
        M.orientation(d)


def test_author_law_exact_contrast():
    rows = [{"focal": t, "x": x} for t in M.J.TARGETS for x in (-1, 1)]
    p, q = M.author_laws(rows, "extraversion")
    score = np.array([r["x"] * (r["focal"] == "extraversion") for r in rows])
    assert (p - q) @ score == pytest.approx(1 / 6)
    assert min(p) > 0 and min(q) > 0
    with pytest.raises(ValueError, match="balanced"):
        M.author_laws(rows[:-1], "anthropomorphism")


def test_cold_function_does_not_require_target_identity():
    obs = {
        "observer_id": "fixed",
        "grain_center": [0] * 4,
        "grain_scale": [1] * 4,
        "query_center": [0] * 6,
        "query_scale": [1] * 6,
        "feature_center": [0] * 49,
        "feature_scale": [1] * 49,
    }
    functions = {
        "test": {
            "arm": "marginal10",
            "status": "SOURCE_ORIENTATION_EXISTS",
            "w": [1] * 10,
            "offset": 0,
        },
        "missing": {
            "arm": "marginal10",
            "status": "NO_COMMON_ORIENTATION_ON_DECLARED_SUPPORT",
            "w": None,
        },
    }
    record = {
        "event_id": "new",
        "observer_id": "fixed",
        "grain": [1] * 4,
        "query": [0.5] * 6,
        "condition": None,
    }
    result = M.cold_score({"observation": obs, "functions": functions}, record)
    assert result["functions"]["test"]["value"] == pytest.approx(7)
    assert result["functions"]["missing"]["value"] is None
    with pytest.raises(ValueError, match="numeric"):
        M.cold_score(
            {"observation": obs, "functions": functions},
            record | {"focal": "extraversion"},
        )
