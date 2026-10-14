#!/usr/bin/env python3
"""Cold unit identity, joint information and calibration-boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_cold_relational as M

ROOT = Path(__file__).resolve().parents[1]


def test_complete_material_and_reordering_identity():
    spec = json.loads((ROOT / "configs/grain_math_cold_relational_v1.json").read_text())
    docs = M.documents(spec)
    assert len(docs) == 480
    assert len({d["body"] for d in docs}) == 480
    pairs = {(d["pair_id"], d["corner"]): d for d in docs if d["form"] == "paraphrase"}
    for d in docs:
        first = pairs[d["pair_id"], d["corner"]]
        assert d["events"] == first["events"]
        assert sorted(d["body"].splitlines()) == sorted(first["body"].splitlines())
        assert d["events"][0]["actor"] == "A"
        assert d["events"][1]["actor"] != d["events"][2]["actor"]
        if d["form"] == "reordered":
            assert d["body"].splitlines() == first["body"].splitlines()[::-1]


def test_missing_target_is_not_a_successful_subset():
    spec = json.loads((ROOT / "configs/grain_math_cold_relational_v1.json").read_text())
    spec["behaviours"].pop("extraversion")
    with pytest.raises(ValueError, match="six"):
        M.documents(spec)


def test_cold_map_is_frozen_not_refitted():
    f = np.array([[1.0, 3.0], [2.0, 4.0]])
    c = np.array([[2.0], [3.0], [-1.0]])
    assert np.array_equal(M.cold_chart(f, c), [[2.0], [4.0]])
    assert np.array_equal(c, [[2.0], [3.0], [-1.0]])
    with pytest.raises(ValueError, match="shape"):
        M.cold_chart(f, np.ones((2, 1)))


def test_independent_pairing_cannot_manufacture_joint_information():
    z = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p = np.array([[0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]])
    delta = M.contrast(p, z)
    assert np.array_equal(delta["means"], [0, 0])
    assert np.array_equal(delta["independent_second"], np.zeros((2, 2)))
    assert np.linalg.norm(delta["second"]) > 0
    d = p[1] - p[0]
    assert d @ M.polynomial_kernel(z) @ d == pytest.approx(
        np.linalg.norm(M.feature_delta(delta)) ** 2
    )


def test_full_tensor_distance_is_rotation_invariant():
    rng = np.random.default_rng(71)
    z = rng.normal(size=(8, 5))
    p = rng.dirichlet(np.ones(8), size=2)
    q = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    a, b = M.contrast(p, z), M.contrast(p, z @ q)
    assert np.linalg.norm(M.feature_delta(a)) == pytest.approx(
        np.linalg.norm(M.feature_delta(b))
    )
    assert np.allclose(M.polynomial_kernel(z), M.polynomial_kernel(z @ q))
    assert np.allclose(b["second"], q.T @ a["second"] @ q)


def test_perturbation_bound_on_nonzero_source_and_cold():
    rng = np.random.default_rng(7102)
    for scale in (0.001, 0.1, 1.0):
        z = rng.normal(size=(8, 5))
        cold = z + rng.normal(size=z.shape) * scale
        p = rng.dirichlet(np.ones(8), size=2)
        result = M.perturbation_certificate(p, z, cold)
        assert result["mean_discrepancy"] <= result["mean_bound"] + 1e-12
        assert result["second_discrepancy"] <= result["second_bound"] + 1e-12


def test_wrong_event_mapping_can_change_author_relation():
    z = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p = np.array([[0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]])
    a = M.feature_delta(M.contrast(p, z))
    wrong = M.feature_delta(M.contrast(p, np.roll(z, 1, axis=0)))
    assert M.cosine(a, wrong) == pytest.approx(0.0)
    assert not np.allclose(a, wrong)


@pytest.mark.parametrize(
    "p", [np.array([[-0.1, 1.1]]), np.array([[0.1, 0.1]]), np.ones((2, 3))]
)
def test_invalid_profiles_fail(p):
    with pytest.raises(ValueError):
        M.moments(p, np.ones((2, 1)))


def test_nonfinite_inputs_fail():
    with pytest.raises(ValueError):
        M.cold_chart([[np.nan]], [[1.0], [1.0]])
    with pytest.raises(ValueError):
        M.perturbation_certificate(
            np.ones((2, 8)) / 8, np.ones((8, 2)), np.ones((8, 3))
        )
    assert M.cosine([0, 0], [1, 1]) is None


def test_second_moments_not_globally_characteristic():
    z = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    # Different distributions, same zero mean and variance1.
    p = np.array([[0, 0.5, 0, 0.5, 0], [0.125, 0, 0.75, 0, 0.125]])
    delta = M.feature_delta(M.contrast(p, z))
    assert np.allclose(delta, 0)
    assert not np.array_equal(p[0], p[1])
